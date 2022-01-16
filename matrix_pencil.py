import numpy as np
import scipy.linalg as la
import scipy.signal as sig

"""
This code was obtained (as I remember) from the github of one of the (reference) authors earlier during my undergraduate studies.
However I took the essentials and modified it for the current framework.
If any information is known about the original source code, please contact me so I can properly cite the github repo.

Reference:
Jacobson, K.E., Kiviaho, J.F., Kennedy, G.J. and Smith, M.J., 2019. 
Evaluation of time-domain damping identification methods for flutter-constrained optimization. Journal of Fluids and Structures, 87, pp.174-188.
"""

class MatrixPencil:
    def __init__(self, t, x, plot=False, rho=100):

        self.X = x
        self.N = len(self.X)
        self.dt = t[1] - t[0]
        self.H = np.eye(self.N)

        # Set the pencil parameter L
        self.n = self.X.shape[0]
        self.L = (self.N / 2) - 1
        self.L = int(self.L)

        # Save model order
        self.M = None

        # Save SVD of Hankel matrix
        self.U = None
        self.s = None
        self.VT = None

        # Save the filtered right singular vectors V1 and the pseudoinverse
        self.V1T = None
        self.V1inv = None

        # Save right singular vectors V2
        self.V2T = None

        # Save eigenvalues and eigenvectors of A matrix
        self.lam = None
        self.W = None
        self.V = None

        # Save amplitudes, damping, frequencies, and phases
        self.amps = None
        self.damp = None
        self.freq = None
        self.faze = None

        # Maximum Damping Coefficient
        self.maxIdx = 0
        self.maxDamp = 0
        self.maxDC = 0
        self.maxMode = {}

        self.ComputeDampingAndFrequency()
        self.ComputeAmplitudeAndPhase()
        self.MaxDampingCoefficient()

    def ComputeDampingAndFrequency(self):
        """-
        Compute the damping and frequencies of the complex exponentials

        """
        # Assemble the Hankel matrix Y from the samples X
        Y = np.empty((self.N - self.L, self.L + 1), dtype=self.X.dtype)
        for i in range(self.N - self.L):
            for j in range(self.L + 1):
                Y[i, j] = self.X[i + j]

        # Compute the SVD of the Hankel matrix
        self.U, self.s, self.VT = la.svd(Y)

        # Estimate the modal order M based on the singular values
        self.EstimateModelOrder()
        # self.M = 4

        # Filter the right singular vectors of the Hankel matrix based on M
        Vhat = self.VT[:self.M, :]
        self.V1T = Vhat[:, :-1]
        self.V2T = Vhat[:, 1:]

        # Compute the pseudoinverse of V1T
        self.V1inv = la.pinv(self.V1T)

        # Form A matrix from V1T and V2T to reduce to eigenvalue problem
        # Jan's method -- recorded in the original Aviation paper
        A = self.V1inv.dot(self.V2T)

        # GJK method - should only be an M x M system. No eigenvalues
        # of very small value.
        # A = self.V2T.dot(self.V1inv)

        # Solve eigenvalue problem to obtain poles
        self.lam, self.W, self.V = la.eig(A, left=True, right=True)

        # Compute damping and freqency
        s = np.log(self.lam[:self.M]) / self.dt
        self.damp = s.real
        self.freq = s.imag

        # force the damping of zero frequency mode to something that
        # won't affect the KS aggregation
        for i in range(self.damp.size):
            if abs(self.freq[i]) < 1e-7:
                self.damp[i] = 0

        return

    def ComputeAmplitudeAndPhase(self):
        """
        Compute the amplitudes and phases of the complex exponentials

        """
        # Compute the residues
        B = np.zeros((self.N, self.M), dtype=self.lam.dtype)
        for i in range(self.N):
            for j in range(self.M):
                B[i, j] = np.abs(self.lam[j]) ** i * np.exp(1.0j * i * np.angle(self.lam[j]))

        r, _, _, _ = la.lstsq(B, self.X)

        # Extract amplitudes and phases from residues
        self.amps = np.abs(r)
        self.faze = np.angle(r)

        return

    def EstimateModelOrder(self):
        """
        Store estimate of model order based on input singular values
        """
        # Normalize the singular values by the maximum and cut out modes
        # corresponding to singular values below a specified tolerance
        tol1 = 1.0e-1
        # print('Tolerance: {}'.format(tol1))
        snorm = self.s / self.s.max()
        n_above_tol = len(self.s[snorm > tol1])

        # Approximate second derivative singular values using convolve as a
        # central difference operator
        w = [1.0, -1.0]
        diff = sig.convolve(snorm, w, 'valid')
        diffdiff = sig.convolve(diff, w, 'valid')

        # Cut out more modes depending on the approximated second derivative
        # The idea is sort of to cut at an inflection point in the singular
        # value curve or maybe where they start to bottom out
        tol2 = tol1
        n_bottom_out = 2 + len(diffdiff[diffdiff > tol2])

        # Estimate the number of modes (model order) to have at least two but
        # otherwise informed by the cuts made above
        self.M = min(max(2, min(n_above_tol, n_bottom_out)), self.L)

        return

    def AggregateDamping(self):
        """
        Kreisselmeier-Steinhauser (KS) function to approximate maximum real
        part of exponent

        Returns
        -------
        float
            approximate maximum of real part of exponents

        """
        m = self.damp.max()
        c = m + np.log(np.sum(np.exp(self.rho * (self.damp - m)))) / self.rho
        return c

    def ReconstructSignal(self, t):
        """
        Having computed the amplitudes, damping, frequencies, and phases, can
        produce signal based on sum of series of complex exponentials

        Parameters
        ----------
        t : numpy.ndarray
            array of times

        Returns
        -------
        X : numpy.ndarray
            reconstructed signal

        """
        X = np.zeros(t.shape)
        for i in range(self.M):
            a = self.amps[i]
            x = self.damp[i]
            w = self.freq[i]
            p = self.faze[i]

            X += a * np.exp(x * (t - self.t0)) * np.cos(w * (t - self.t0) + p)

        return X

    def MaxDampingCoefficient(self):
        self.maxIdx = np.argmax(np.abs(self.damp))
        self.maxMode = {'Amplitude': self.amps[self.maxIdx], 'Growth': self.damp[self.maxIdx],
                        'Frequency': self.freq[self.maxIdx], 'Phase': self.faze[self.maxIdx]}
        self.maxDamp = self.damp[self.maxIdx]

        self.maxDC = self.damp[self.maxIdx] / np.abs(self.freq[self.maxIdx])

""" Function to choose the greatest damping coefficient """
def compare(pitchMode : MatrixPencil, plungeMode : MatrixPencil) -> float:
    max_pitch_damp = pitchMode.damp[np.argmax(np.abs(pitchMode.damp))]
    max_plunge_damp = plungeMode.damp[np.argmax(np.abs(plungeMode.damp))]
    max_dc = pitchMode.damp[np.argmax(np.abs(pitchMode.damp))] / np.abs(
        pitchMode.freq[np.argmax(np.abs(pitchMode.damp))])
    print('Pitch DC = {}'.format(-max_dc))
    
    max_dc = plungeMode.damp[np.argmax(np.abs(plungeMode.damp))] / np.abs(
        plungeMode.freq[np.argmax(np.abs(plungeMode.damp))])
    print('Plunge DC = {}'.format(-max_dc))        
    
    if np.abs(max_pitch_damp) > np.abs(max_plunge_damp):
        max_dc = pitchMode.damp[np.argmax(np.abs(pitchMode.damp))] / np.abs(
            pitchMode.freq[np.argmax(np.abs(pitchMode.damp))])
        # print('Pitch DC')
    elif np.abs(max_plunge_damp) > np.abs(max_pitch_damp):
        max_dc = plungeMode.damp[np.argmax(np.abs(plungeMode.damp))] / np.abs(
            plungeMode.freq[np.argmax(np.abs(plungeMode.damp))])
        # print('Plunge DC')
    return max_dc



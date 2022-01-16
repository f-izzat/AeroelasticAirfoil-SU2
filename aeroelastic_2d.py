import os
import numpy as np
import subprocess
from shutil import copy

from matrix_pencil import MatrixPencil, compare


ABS_PATH_TO_SU2_CFD = "E:/Izzaturrahman/SU2-7.2.1/bin/SU2_CFD.exe"
ABS_PATH_TO_MESH = "E:/Izzaturrahman/AEROELASTIC/MASTER/NACA_64A010.su2"
MESH_FILENAME = ABS_PATH_TO_MESH.split('/')[-1]

class Configuration:
    """
    SU2 Configuration (.cfg) files created here. Current implementation is suited for aeroelastic simulation.
    Feel free to alter for any other case.
    """
    def __init__(self, forced : bool = False) -> None:
        self.forced = forced
        self.cfg : str = ""
        self.mesh_filename : str = ""
        self.time_iter = 0
        self._solver = dict(initializeed = False,
                            params = ["EULER", "NAVIER_STOKES", "RANS", "INC_EULER", "INC_NAVIER_STOKES", "INC_RANS"])

    def solver_setup(self, solver : str, **kwargs):
        # https://su2code.github.io/docs_v7/Solver-Setup/
        self.cfg += "% ------------- DIRECT, ADJOINT, AND LINEARIZED PROBLEM DEFINITION ------------%\n"
        assert solver.upper() in self._solver["params"]
        self.cfg += f"SOLVER = {solver.upper()}" + "\n"
        self.cfg += "SYSTEM_MEASUREMENTS = SI \n"
        self.cfg += "MATH_PROBLEM = DIRECT \n"
        self.cfg += "READ_BINARY_RESTART = NO \n"
        if not self.forced:
            assert "restart_iter" in kwargs
            self.cfg += "RESTART_SOL = YES \n"
            self.cfg += f"RESTART_ITER = {kwargs['restart_iter']}" + "\n"
        
        self._solver["initialized"] = True

    def unsteady(self, time_step : float, max_time : float, **kwargs):
        # https://su2code.github.io/docs_v7/Solver-Setup/
        self.cfg += "% ------------------------ Time-dependent Simulation --------------------------%\n"
        self.cfg += "TIME_DOMAIN = YES\n" \
                  + f"TIME_STEP = {time_step}" + "\n" \
                  + f"MAX_TIME = {max_time}" + "\n"
        if "inner_iter" in kwargs:
            self.cfg += f"INNER_ITER = {kwargs['inner_iter']}" + "\n"
        if "time_iter" in kwargs:
            self.cfg += f"TIME_ITER = {kwargs['time_iter']}" + "\n"
            self.time_iter = kwargs['time_iter']
        else:
            self.cfg += f"TIME_ITER = {int(max_time/time_step)}" + "\n"
            self.time_iter = int(max_time/time_step)
        if "time_marching" in kwargs:
            self.cfg += f"TIME_MARCHING = {kwargs['time_marching']}" + "\n"        

    def freestream(self, compressible : bool, mach : float, **kwargs):
        # https://su2code.github.io/docs_v7/Physical-Definition/
        assert self._solver["initialized"]
        self.cfg += f"MACH_NUMBER = {mach}" + "\n" 
        self.cfg += f"MACH_MOTION = {mach}" + "\n" 
        if compressible:
            self.cfg += "% -------------------- COMPRESSIBLE FREE-STREAM DEFINITION --------------------%\n"
            if "aoa" in kwargs:
                self.cfg += f"AOA = {kwargs.get('aoa')}" + "\n"
            else:
                self.cfg += "AOA = 0.0\n"
            if "pressure" in kwargs:
                self.cfg += f"FREESTREAM_PRESSURE = {kwargs.get('pressure')}" + "\n"
            else:
                self.cfg += "FREESTREAM_PRESSURE = 101325.0\n"                
            if "temperature" in kwargs:
                self.cfg += f"FREESTREAM_TEMPERATURE = {kwargs.get('temperature')}" + "\n"
            else:
                self.cfg += "FREESTREAM_TEMPERATURE = 288.15\n"
            if "density" in kwargs:
                self.cfg += f"FREESTREAM_DENSITY = {kwargs.get('density')}" + "\n"
        else:
            self.cfg += "% ------------------- INCOMPRESSIBLE FREE-STREAM DEFINITION -------------------%\n"
            # Density Model
            if "density_model" in kwargs:
                assert kwargs["density_model"] in ["BOUSSINESQ", "VARIABLE"]
                self.cfg += f"INC_DENSITY_MODEL = {kwargs.get('density_model')}" + "\n"
            else:
                self.cfg += "INC_DENSITY_MODEL = CONSTANT" + "\n"
            # Energy Equation
            if "energy_equation" in kwargs:
                assert kwargs["energy_equation"].upper() in ["YES", "NO"]
                self.cfg += f"INC_ENERGY_EQUATION = {kwargs.get('energy_equation')}" + "\n"
            # Inlet Type
            if "inlet_type" in kwargs:
                assert kwargs["inlet_type"].upper() in ["VELOCITY_INLET", "PRESSURE_INLET"]
                self.cfg += f"INC_INLET_TYPE = {kwargs.get('inlet_type')}" + "\n"
            # Initial Values
            if "init_density" in kwargs:
                self.cfg += f"INC_DENSITY_INIT = {kwargs.get('init_density')}" + "\n"
            else:
                self.cfg += "INC_DENSITY_INIT = 1.2886 \n"
            if "init_temperature" in kwargs:
                self.cfg += f"INC_TEMPERATURE_INIT = {kwargs.get('init_temperature')}" + "\n"
            else:
                self.cfg += "INC_TEMPERATURE_INIT = 288.15 \n"                
            if "init_velocity" in kwargs:
                assert isinstance(kwargs.get('velocity'), tuple)
                assert len(kwargs.get('velocity')) == 3
                self.cfg += f"INC_VELOCITY_INIT = {kwargs.get('init_velocity')}" + "\n"
            else:
                self.cfg += "INC_VELOCITY_INIT = (0.0, 0.0, 0.0) \n"     
            # Reference Values
            if "ref_density" in kwargs:
                self.cfg += f"INC_DENSITY_REF = {kwargs.get('ref_density')}" + "\n"          
            if "ref_velocity" in kwargs:
                self.cfg += f"INC_VELOCITY_REF = {kwargs.get('ref_velocity')}" + "\n"
            if "ref_temperature" in kwargs:
                self.cfg += f"INC_TEMPERATURE_REF = {kwargs.get('ref_temperature')}" + "\n"
               
    def reference(self, X : float, Y : float, Z : float, length : float = 1.0, area : float = 1.0, **kwargs):
        self.cfg += "% ---------------------- REFERENCE VALUE DEFINITION ---------------------------%\n"
        self.cfg += f"REF_ORIGIN_MOMENT_X = {X}" + "\n" \
                  + f"REF_ORIGIN_MOMENT_Y = {Y}" + "\n" \
                  + f"REF_ORIGIN_MOMENT_Z = {Z}" + "\n" \
                  + f"REF_LENGTH = {length}" + "\n" \
                  + f"REF_AREA = {area}" + "\n"
        if "non_dimensionalization" in kwargs:
            assert kwargs["non_dimensionalization"].upper() in \
                    ["DIMENSIONAL", "FREESTREAM_PRESS_EQ_ONE", "FREESTREAM_VEL_EQ_MACH", "FREESTREAM_VEL_EQ_ONE"]
            self.cfg += f"REF_DIMENSIONALIZATION = {kwargs.get('non_dimensionalization').upper()}" + "\n"                                     
        else:
            self.cfg += "REF_DIMENSIONALIZATION = DIMENSIONAL \n"

    def dynamic_mesh(self, **kwargs):
        self.cfg += "% ----------------------- DYNAMIC MESH DEFINITION -----------------------------%\n"
        if self.forced:
            self.cfg += "GRID_MOVEMENT = RIGID_MOTION \n"
        else:
            self.cfg += "SURFACE_MOVEMENT = AEROELASTIC \n"
            assert 'marker_moving' in kwargs            
            self.cfg += f"MARKER_MOVING = ({kwargs['marker_moving']})" + "\n"        
        if 'pitching_omega' in kwargs:
            assert len(kwargs['pitching_omega']) == 3
            self.cfg += f"PITCHING_OMEGA = {kwargs['pitching_omega'][0]} {kwargs['pitching_omega'][1]} {kwargs['pitching_omega'][2]}" + "\n"
        if 'pitching_amplitude' in kwargs:
            assert len(kwargs['pitching_amplitude']) == 3
            self.cfg += f"PITCHING_AMPL = {kwargs['pitching_amplitude'][0]} {kwargs['pitching_amplitude'][1]} {kwargs['pitching_amplitude'][2]}" + "\n"        
        if 'motion_origin' in kwargs:
            assert len(kwargs['motion_origin']) == 3
            self.cfg += f"MOTION_ORIGIN = {kwargs['motion_origin'][0]} {kwargs['motion_origin'][1]} {kwargs['motion_origin'][2]}" + "\n"            

    def boundary_condition(self, **kwargs):
        self.cfg += "% -------------------- BOUNDARY CONDITION DEFINITION --------------------------%\n"
        if "marker_euler" in kwargs:
            self.cfg += f"MARKER_EULER = ({kwargs.get('marker_euler')})" + "\n"
        if "marker_far" in kwargs:
            self.cfg += f"MARKER_FAR = ({kwargs.get('marker_far')})" + "\n"

    def surfaces_def(self, **kwargs):
        self.cfg += "% ------------------------ SURFACES IDENTIFICATION ----------------------------%\n"
        if "marker_plotting" in kwargs:
            self.cfg += f"MARKER_PLOTTING = ({kwargs.get('marker_plotting')})" + "\n"
        if "marker_monitoring" in kwargs:
            self.cfg += f"MARKER_MONITORING = ({kwargs.get('marker_monitoring')})" + "\n"            

    def linear_solver(self, **kwargs):
        self.cfg += "% ------------------------ LINEAR SOLVER DEFINITION ---------------------------%\n"
        if "solver" in kwargs:
            self.cfg += f"LINEAR_SOLVER = {kwargs.get('solver')}" + "\n"
        if "prec" in kwargs:
            self.cfg += f"LINEAR_SOLVER_PREC = {kwargs.get('prec')}" + "\n"
        if "error" in kwargs:
            self.cfg += f"LINEAR_SOLVER_ERROR = {kwargs.get('error')}" + "\n"
        if "iter" in kwargs:
            self.cfg += f"LINEAR_SOLVER_ITER = {kwargs.get('iter')}" + "\n"                           

    def multigrid(self, **kwargs):
        self.cfg += "% -------------------------- MULTIGRID PARAMETERS -----------------------------%\n"
        if "level" in kwargs:
            self.cfg += f"MGLEVEL = {kwargs.get('level')}" + "\n"
        if "cycle" in kwargs:
            self.cfg += f"MGCYCLE = {kwargs.get('cycle')}" + "\n"
        if "pre_smooth" in kwargs:
            assert isinstance(kwargs['pre_smooth'], tuple)
            self.cfg += f"MG_PRE_SMOOTH = {kwargs.get('pre_smooth')}" + "\n"            
        if "post_smooth" in kwargs:
            assert isinstance(kwargs['post_smooth'], tuple)
            self.cfg += f"MG_POST_SMOOTH = {kwargs.get('post_smooth')}" + "\n"    
        if "correction_smooth" in kwargs:
            assert isinstance(kwargs['correction_smooth'], tuple)
            self.cfg += f"MG_CORRECTION_SMOOTH = {kwargs.get('correction_smooth')}" + "\n"
        if "damp_restriction" in kwargs:
            self.cfg += f"MG_DAMP_RESTRICTION = {kwargs.get('damp_restriction')}" + "\n"           
        if "damp_prolongation" in kwargs:
            self.cfg += f"MG_DAMP_PROLONGATION = {kwargs.get('damp_prolongation')}" + "\n"                                

    def numerical_method(self, **kwargs):
        self.cfg += "% ----------------------- NUMERICAL METHOD DEFINITION -------------------------%\n"
        if "grad" in kwargs:
            self.cfg += f"NUM_METHOD_GRAD = {kwargs.get('grad')}" + "\n"
        if "cfl" in kwargs:
            self.cfg += f"CFL_NUMBER = {kwargs.get('cfl')}" + "\n"              
        if "conv_num_method_flow" in kwargs:
            self.cfg += f"CONV_NUM_METHOD_FLOW = {kwargs.get('conv_num_method_flow')}" + "\n"              
        if "muscl_flow" in kwargs:
            assert kwargs['muscl_flow'] in ["YES", "NO"]
            self.cfg += f"MUSCL_FLOW = {kwargs.get('muscl_flow')}" + "\n"
        if "slope_limiter_flow" in kwargs:
            self.cfg += f"SLOPE_LIMITER_FLOW = {kwargs.get('slope_limiter_flow')}" + "\n"
        if "entropy_fix_coeff" in kwargs:
            self.cfg += f"ENTROPY_FIX_COEFF = {kwargs.get('entropy_fix_coeff')}" + "\n"
        if "jst_sensor_coeff" in kwargs:
            assert isinstance(kwargs['jst_sensor_coeff'], tuple)
            self.cfg += f"JST_SENSOR_COEFF = {kwargs.get('jst_sensor_coeff')}" + "\n"
        if "time_discre_flow" in kwargs:
            self.cfg += f"TIME_DISCRE_FLOW = {kwargs.get('time_discre_flow')}" + "\n"                                                         

    def grid_deformation(self, **kwargs):
        self.cfg += "% ------------------------ GRID DEFORMATION PARAMETERS ------------------------%\n"
        if "linear_solver" in kwargs:
            self.cfg += f"DEFORM_LINEAR_SOLVER = {kwargs.get('linear_solver')}" + "\n"
        if "linear_solver_prec" in kwargs:
            self.cfg += f"DEFORM_LINEAR_SOLVER_PREC = {kwargs.get('linear_solver_prec')}" + "\n"
        if "linear_solver_iter" in kwargs:
            self.cfg += f"DEFORM_LINEAR_SOLVER_ITER = {kwargs.get('linear_solver_iter')}" + "\n"
        if "linear_solver_error" in kwargs:
            self.cfg += f"DEFORM_LINEAR_SOLVER_ERROR = {kwargs.get('linear_solver_error')}" + "\n"            
        if "nonliner_iter" in kwargs:
            self.cfg += f"DEFORM_NONLINEAR_ITER = {kwargs.get('nonliner_iter')}" + "\n"
        if "stiffness_type" in kwargs:
            self.cfg += f"DEFORM_STIFFNESS_TYPE = {kwargs.get('stiffness_type')}" + "\n"            

    def convergence(self, **kwargs):
        self.cfg += "% --------------------------- CONVERGENCE PARAMETERS --------------------------%\n"
        if "criteria" in kwargs:
            self.cfg += f"CONV_CRITERIA = {kwargs.get('criteria')}" + "\n"
        if "residual_minval" in kwargs:
            self.cfg += f"CONV_RESIDUAL_MINVAL = {kwargs.get('residual_minval')}" + "\n"
        if "start_iter" in kwargs:
            self.cfg += f"CONV_STARTITER = {kwargs.get('start_iter')}" + "\n"
        if "cauchy_elems" in kwargs:
            self.cfg += f"CONV_CAUCHY_ELEMS = {kwargs.get('cauchy_elems')}" + "\n"
        if "cauchy_eps" in kwargs:
            self.cfg += f"CONV_CAUCHY_EPS = {kwargs.get('cauchy_eps')}" + "\n"                                                         

    def input_output(self, mesh_filename : str, **kwargs):
        self.cfg += "% ------------------------- INPUT/OUTPUT INFORMATION --------------------------%\n"
        self.mesh_filename = mesh_filename
        self.cfg += f"MESH_FILENAME = {mesh_filename}" + "\n"
        if "mesh_format" in kwargs:
            self.cfg += f"MESH_FORMAT = {kwargs.get('mesh_format')}" + "\n"                     
        else:
            self.cfg += "MESH_FORMAT = SU2 \n"
        if "solution_filename" in kwargs:
            self.cfg += f"SOLUTION_FILENAME = {kwargs.get('solution_filename')}" + "\n"
        else:
            self.cfg += "SOLUTION_FILENAME = solution.dat \n"
        if "tabular_format" in kwargs:
            self.cfg += f"TABULAR_FORMAT = {kwargs.get('tabular_format')}" + "\n"
        else:
            self.cfg += "TABULAR_FORMAT = CSV \n"
        if "conv_filename" in kwargs:
            self.cfg += f"CONV_FILENAME = {kwargs.get('conv_filename')}" + "\n"
        else:
            self.cfg += "CONV_FILENAME = history \n"
        if "restart_filename" in kwargs:
            self.cfg += f"RESTART_FILENAME = {kwargs.get('restart_filename')}" + "\n"
        else:
            self.cfg += "RESTART_FILENAME = restart.csv \n"
        if "volume_filename" in kwargs:
            self.cfg += f"VOLUME_FILENAME = {kwargs.get('volume_filename')}" + "\n"
        else:
            self.cfg += "VOLUME_FILENAME = volume \n"
        if "surface_filename" in kwargs:
            self.cfg += f"SURFACE_FILENAME = {kwargs.get('surface_filename')}" + "\n"
        else:
            self.cfg += "SURFACE_FILENAME = surface \n"
        if "output_wrt_freq" in kwargs:
            self.cfg += f"OUTPUT_WRT_FREQ = {kwargs.get('output_wrt_freq')}" + "\n"
        else:
            self.cfg += "OUTPUT_WRT_FREQ = 1 \n"

        if "screen_output" in kwargs:
            assert isinstance(kwargs['screen_output'], tuple)
            self.cfg += "SCREEN_OUTPUT = ("
            for so in range(len(kwargs['screen_output'])):
                if so != len(kwargs['screen_output']) - 1:
                    self.cfg += kwargs['screen_output'][so] + ","
                else:
                    self.cfg += kwargs['screen_output'][so] + ") \n"

        if "history_output" in kwargs:
            assert isinstance(kwargs['history_output'], tuple)
            self.cfg += "HISTORY_OUTPUT = ("
            for ho in range(len(kwargs['history_output'])):
                if ho != len(kwargs['history_output']) - 1:
                    self.cfg += kwargs['history_output'][ho] + ","
                else:
                    self.cfg += kwargs['history_output'][ho] + ") \n"

        # HISTORY_WRT_FREQ_INNER, HISTORY_WRT_FREQ_OUTER and HISTORY_WRT_FREQ_TIME
        if "history_wrt_freq_inner" in kwargs:
            self.cfg += f"HISTORY_WRT_FREQ_INNER = {kwargs.get('history_wrt_freq_inner')}" + "\n"
        else:            
            self.cfg += "HISTORY_WRT_FREQ_INNER = 0 \n"            
        if "history_wrt_freq_outer" in kwargs:
            self.cfg += f"HISTORY_WRT_FREQ_OUTER = {kwargs.get('history_wrt_freq_outer')}" + "\n"
        else:            
            self.cfg += "HISTORY_WRT_FREQ_OUTER = 0 \n"
        if "history_wrt_freq_time" in kwargs:
            self.cfg += f"HISTORY_WRT_FREQ_TIME = {kwargs.get('history_wrt_freq_time')}" + "\n"
        else:                            
            self.cfg += "HISTORY_WRT_FREQ_TIME = 1 \n"

        # SCREEN_WRT_FREQ_INNER, SCREEN_WRT_FREQ_OUTER, SCREEN_WRT_FREQ_TIME
        if "screen_wrt_freq_inner" in kwargs:
            self.cfg += f"SCREEN_WRT_FREQ_INNER = {kwargs.get('screen_wrt_freq_inner')}" + "\n"
        else:            
            self.cfg += "SCREEN_WRT_FREQ_INNER = 1 \n"            
        if "screen_wrt_freq_outer" in kwargs:
            self.cfg += f"SCREEN_WRT_FREQ_OUTER = {kwargs.get('screen_wrt_freq_outer')}" + "\n"
        else:            
            self.cfg += "SCREEN_WRT_FREQ_OUTER = 1 \n"
        if "screen_wrt_freq_time" in kwargs:
            self.cfg += f"SCREEN_WRT_FREQ_TIME = {kwargs.get('screen_wrt_freq_time')}" + "\n"
        else:                            
            self.cfg += "SCREEN_WRT_FREQ_TIME = 1 \n"

        self.cfg += "OUTPUT_FILES = (RESTART_ASCII) \n"                                                                    

    def aeroelastic(self, plunge_nat_freq: float, pitch_nat_freq: float, 
                    mass_ratio: float, cg_location: float, radius_gyration: float, 
                    flutter_speed_index : float,
                    iterations: int):        
        assert not self.forced
        self.cfg += "% -------------- AEROELASTIC SIMULATION (Typical Section Model) ---------------%\n"
        self.cfg += f"PLUNGE_NATURAL_FREQUENCY = {plunge_nat_freq}" + "\n" \
                  + f"PITCH_NATURAL_FREQUENCY = {pitch_nat_freq}" + "\n" \
                  + f"AIRFOIL_MASS_RATIO = {mass_ratio}" + "\n" \
                  + f"CG_LOCATION = {cg_location}" + "\n" \
                  + f"RADIUS_GYRATION_SQUARED = {radius_gyration}" + "\n" \
                  + f"FLUTTER_SPEED_INDEX = {flutter_speed_index}" + "\n" \
                  + f"AEROELASTIC_ITER = {iterations}" + "\n"

class Aeroelastic:
    """
    Aeroelastic simulation takes place in two phases, a forced vibration to initialize followed by a free vibration
    Args:
        path_to_exe    : str = Absolute path to SU2_CFD(.exe)
        working_folder : str = Absolute path to a (non)existent folder where solutions (i.e. restart_flow.csv) will be stored
        forced         : Configuration = Forced configuration object
        free           : Configuration = Free configuration object
    """
    def __init__(self, path_to_exe : str, working_folder : str, forced : Configuration, free : Configuration) -> None:
        self.forced_cfg = forced 
        self.free_cfg = free
        self.path_to_exe = path_to_exe
        assert os.path.exists(f"{working_folder}/{forced.mesh_filename}"), f"Cant find {working_folder}/{forced.mesh_filename}"

        with open(f"{working_folder}/forced.cfg", 'w') as ff:
            ff.write(forced.cfg)
        with open(f"{working_folder}/free.cfg", 'w') as ff:
            ff.write(free.cfg)            
        
        self.working_folder = working_folder
        self.damping_coefficient = 0
        self.m_forced = False
        self.m_free = False

    def forced(self, n_parallel = 1) -> None:
        self.m_forced = True
        cmd = ["mpiexec", "-n", f"{n_parallel}", self.path_to_exe, "forced.cfg"] if n_parallel > 1 else [self.path_to_exe, "forced.cfg"]
        subprocess.run(cmd, shell=True, cwd=self.working_folder)

    def free(self, n_parallel = 1) -> None:
        assert self.m_forced
        self.m_free = True
        cmd = ["mpiexec", "-n", f"{n_parallel}", self.path_to_exe, "free.cfg"] if n_parallel > 1 else [self.path_to_exe, "free.cfg"]
        subprocess.run(cmd, shell=True, cwd=self.working_folder)

    def compute_damping_coefficient(self):
        assert self.m_free
        data = np.genfromtxt(f"{self.working_folder}/history_000{self.forced_cfg.time_iter}.csv", dtype=None, delimiter=',')
        for idx, ss in enumerate(data[0, :]):
            if str(ss).split(r'"')[1] == "Cur_Time":
                time = np.array(data[1:, idx], dtype=np.float64)
            if str(ss).split(r'"')[1] == "plunge(airfoil)":
                plunge = np.array(data[1:, idx], dtype=np.float64)
            if str(ss).split(r'"')[1] == "pitch(airfoil)":
                pitch = np.array(data[1:, idx], dtype=np.float64)

        plunge_mp = MatrixPencil(t=time, x=plunge)
        pitch_mp = MatrixPencil(t=time, x=pitch)
        self.damping_coefficient = compare(pitchMode=pitch_mp, plungeMode=plunge_mp)

    def __call__(self, n_parallel = 1) -> None:
        self.forced(n_parallel)
        self.free(n_parallel)
        self.compute_damping_coefficient()

"""
- Function was written to accomodate for multiple flow conditions i.e. varying Mach and Flutter Speed Indices.
- Which is why there is a single (master) mesh file i.e. PATH_TO_MESH.
- The absolute path to a (non-)existent working folder is given so as to store the solution of a single run.
"""
def NACA_64A010(working_folder_path : str, mach : float, flutter_speed_index : float, n_parallel : int = 1) -> Aeroelastic:
    """ DEFINE AEROELASTIC SYSTEM """
    omega_theta = 100 # rad/s
    Xref = -0.5
    # Distance in semichords by which the center of gravity lies behind the elastic axis
    cg_location = 1.8
    # The radius of gyration squared (expressed in semichords) of the typical section about the elastic axis
    radius_gyration = 3.48
    # Airfoil Mass Ratio
    miu = 60    

    """ DEFINE FLOW PROPERTIES """
    gamma = 1.4
    R = 287    
    freestream_temp = ((flutter_speed_index * omega_theta * Xref * np.sqrt(miu)) / mach)**2 / (gamma * R) 

    """ INITIALIZE SOLUTION -> Create forced.cfg """
    # 36 steps per period, based on the pitch natural frequency
    time_step = 2*np.pi / (omega_theta * 36) 
    # Iterate for a total of forced_iter iterations
    forced_iter = 72
    forced_max_time = time_step * forced_iter

    forced = Configuration(forced = True)
    forced.solver_setup(solver = "EULER")    
    forced.unsteady(
        time_marching = "DUAL_TIME_STEPPING-2ND_ORDER", 
        time_step = time_step, 
        max_time = forced_max_time, 
        inner_iter = 251
    )
    forced.freestream(
        compressible = True, mach = mach,
        aoa = 0.0, temperature = freestream_temp
    )
    forced.reference(
        X = Xref, Y = 0.0, Z = 0.0,
        length = 1.0, area = 1.0
    )
    forced.dynamic_mesh(
        motion_origin = (-0.5, 0.0, 0.0),
        pitching_omega = (0.0, 0.0, 100.0),
        pitching_amplitude = (0.0, 0.0, 1.0)
    )
    forced.boundary_condition(
        marker_euler = "airfoil",
        marker_far = "farfield"
    )
    forced.surfaces_def(
        marker_plotting = "airfoil",
        marker_monitoring = "airfoil",
    )
    forced.linear_solver(
        solver = "FGMRES",
        prec = "LU_SGS",
        error = 1E-4,
        iter = 2
    )
    forced.multigrid(
        level = 3,
        cycle = "W_CYCLE",
        pre_smooth = (1, 2, 3, 3),
        post_smooth = (0, 0, 0, 0),
        correction_smooth = (0, 0, 0, 0),
        damp_restriction = 0.75,
        damp_prolongation = 0.75
    )
    forced.numerical_method(
        grad = "WEIGHTED_LEAST_SQUARES",
        cfl = 10.0,
        conv_num_method_flow = "JST",
        muscl_flow = "YES",
        slope_limiter_flow = "VENKATAKRISHNAN",
        entropy_fix_coeff = 0.001,
        jst_sensor_coeff = (0.5, 0.02),
        time_discre_flow = "EULER_IMPLICIT"
    )
    forced.grid_deformation(
        linear_solver = "FGMRES",
        linear_solver_prec = "LU_SGS",
        linear_solver_iter = 500,
        nonlinear_iter = 1,
        stiffness_type = "INVERSE_VOLUME",
    )
    forced.convergence(
        criteria = "RESIDUAL",
        residual_minval = -8,
        start_iter = 0,
        cauchy_elems = 100,
        cauchy_eps = 1E-10
    )
    forced.input_output(
        mesh_filename = MESH_FILENAME,
        output_wrt_freq = 1,
        screen_output = ("TIME_ITER", "INNER_ITER", "RMS_DENSITY", "RMS_ENERGY", "LIFT", "DRAG_ON_SURFACE"),
        history_output = ("ITER", "RMS_RES", "AERO_COEFF", "TIME_DOMAIN", "WALL_TIME")  
    )

    """ UNSTEADY SOLUTION -> Create free.cfg """
    # Iterate for a total of free_iter iterations
    free_iter = 202
    free_max_time = time_step * free_iter
    free = Configuration()
    free.solver_setup(solver = "EULER", restart_iter = forced_iter)
    free.unsteady(
        time_marching = "DUAL_TIME_STEPPING-2ND_ORDER", 
        time_step = time_step, 
        max_time = free_max_time, 
        inner_iter = 100        
    )
    free.freestream(
        compressible = True, mach = mach,
        aoa = 0.0, temperature = freestream_temp
    )
    free.reference(
        X = Xref, Y = 0.0, Z = 0.0,
        length = 1.0, area = 1.0
    ) 
    free.dynamic_mesh(
        marker_moving = "airfoil"
    )    
    free.boundary_condition(
        marker_euler = "airfoil",
        marker_far = "farfield"
    )   
    free.surfaces_def(
        marker_plotting = "airfoil",
        marker_monitoring = "airfoil",
    )
    free.linear_solver(
        solver = "FGMRES",
        prec = "LU_SGS",
        error = 1E-4,
        iter = 2
    )
    free.multigrid(
        level = 3,
        cycle = "W_CYCLE",
        pre_smooth = (1, 2, 3, 3),
        free_smooth = (0, 0, 0, 0),
        correction_smooth = (0, 0, 0, 0),
        damp_restriction = 0.75,
        damp_prolongation = 0.75
    )
    free.numerical_method(
        grad = "GREEN_GAUSS",
        cfl = 4.0,
        conv_num_method_flow = "JST",
        muscl_flow = "YES",
        slope_limiter_flow = "VENKATAKRISHNAN",
        entropy_fix_coeff = 0.001,
        jst_sensor_coeff = (0.5, 0.02),
        time_discre_flow = "EULER_IMPLICIT"
    )
    free.grid_deformation(
        linear_solver = "FGMRES",
        linear_solver_prec = "LU_SGS",
        linear_solver_iter = 500,
        nonlinear_iter = 1,
        stiffness_type = "INVERSE_VOLUME",
    )
    free.convergence(
        criteria = "RESIDUAL",
        residual_minval = -8,
        start_iter = 0,
        cauchy_elems = 100,
        cauchy_eps = 1E-10
    )
    free.input_output(
        mesh_filename = MESH_FILENAME,
        solution_filename = "restart.csv",
        output_wrt_freq = 1,
        screen_output = ("TIME_ITER", "INNER_ITER", "RMS_DENSITY", "RMS_ENERGY", "LIFT",  "DRAG", "PITCH", "PLUNGE", "WALL_TIME"),
        history_output = ("ITER", "RMS_RES", "AERO_COEFF", "AEROELASTIC", "TIME_DOMAIN", "WALL_TIME")  
    )
    free.aeroelastic(
        plunge_nat_freq = omega_theta,
        pitch_nat_freq = omega_theta,
        mass_ratio = miu,
        cg_location = cg_location,
        radius_gyration = radius_gyration,
        flutter_speed_index = flutter_speed_index,
        iterations = 3
    )        

    """ 
    CALL SU2 
    path_to_exe : Absolute path to SU2_CFD.exe
    folder      : Absolute path to working folder i.e. folder with .su2 mesh file in, solution files will be stored in here.
    n_parallel  : Args for command 'mpiexec -n {n_parallel}'
    """
    # Create working folder, paste mesh file into folder
    if not os.path.exists(working_folder_path):
        os.mkdir(working_folder_path)
    if not os.path.exists(f"{working_folder_path}/{MESH_FILENAME}"):
        copy(ABS_PATH_TO_MESH, working_folder_path)

    _simulation = Aeroelastic(path_to_exe=ABS_PATH_TO_SU2_CFD,
                             working_folder=working_folder_path, 
                             forced=forced, free=free)
    _simulation(n_parallel=n_parallel)
    return _simulation

if __name__ == "__main__":
    mach = 0.635
    flutter_speed_index = 0.9
    simulation = NACA_64A010(
        mach=mach, flutter_speed_index=flutter_speed_index, n_parallel=3,
        working_folder_path=f"E:/Izzaturrahman/AEROELASTIC/NACA_64A010/M_{mach}_VF_{flutter_speed_index}"
    )
    print(simulation.damping_coefficient)
    # 0.635	0.9	0.062768485


   

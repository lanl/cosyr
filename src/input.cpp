#include "input.h"

namespace cosyr {
using namespace py::literals; // to bring in the `_a` literal for keyword argument

/* -------------------------------------------------------------------------- */
Input::Input(int argc, char **argv, Timer& in_timer) : timer(in_timer) {

  // initialize runtime
  mpi.comm = MPI_COMM_WORLD;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(mpi.comm, &mpi.rank);
  MPI_Comm_size(mpi.comm, &mpi.num_ranks);

  Kokkos::InitArguments args;
  // (CPU) threads per NUMA region
  //args.num_threads = 4;
  // (CPU) NUMA regions per process
  //args.num_numa = 1;
  // GPU device ID 
#define NUM_GPU_PER_NODE 4
  // TODO: this needs to be generalized
  args.device_id = mpi.rank % NUM_GPU_PER_NODE;
  Kokkos::initialize(args);

  if (argc < 2) {
    std::cerr << "Error: wrong number of arguments." << std::endl;
    std::cerr << "Usage: ./cosyr input.py" << std::endl;
    Kokkos::finalize(); //TODO: we can't finalize Kokkos here, need fix
    MPI_Finalize();
    std::exit(EXIT_FAILURE);
  }

  timer.start("process_input_deck");

  // start the interpreter and keep it alive
  auto sys = py::module::import("sys");
  auto py_version = sys.attr("version").cast<std::string>();
  if (mpi.rank == 0) { py::print("Python version " + py_version + " interpreter started"); }

  auto params      = py::array_t<double>(2);
  auto py_beam     = py::array_t<double>(0);
  auto wave_x      = py::array_t<double>(0);
  auto wave_y      = py::array_t<double>(0);
  #if DIM == 3
    auto wave_z    = py::array_t<double>(0);
  #endif
  auto wave_field  = py::array_t<double>(0);
  auto remap_scale = py::array_t<double>(DIM);
  auto coord_shift = py::array_t<double>(DIM);
  auto coord_scale = py::array_t<double>(DIM);
  // load default values for python variables in locals()
  locals = py::dict("COSYR_Version"_a  = COSYR_VER,
                    // kernel
                    "run_name"_a  = kernel.run_name,
                    "num_wavefronts"_a = kernel.num_wavefronts,
                    "num_dirs"_a = kernel.num_dirs,
                    "num_step"_a = kernel.num_step,
                    "num_particles"_a = kernel.num_particles,
                    "qm"_a = kernel.qm,
                    "trajectory_type"_a = static_cast<int>(kernel.trajectory),
                    "emission_start"_a = kernel.emission_start_step,
                    "emission_interval"_a = kernel.emission_interval,                   
                    "print_interval"_a = kernel.print_interval,
                    "min_emit_angle"_a = kernel.min_emit_angle,
                    "dt"_a = kernel.dt,
                    "radius"_a = kernel.radius,
                    "parameters"_a = params,                    
                    //mesh
                    "num_gridpt_hor"_a = mesh.num_hor,
                    "num_gridpt_ver"_a = mesh.num_ver,
                    "mesh_span_angle"_a = mesh.span_angle,
                    "mesh_width"_a = mesh.width,
                    "mesh_output"_a = mesh.output,
                    "mesh_output_start"_a = mesh.dump_start,
                    "mesh_output_interval"_a = mesh.dump_interval,
                    // remap                                        
                    "remap_start"_a = remap.start_step,                    
                    "remap_interval"_a = remap.interval,
                    "remap_scatter"_a = remap.scatter,
                    "remap_adaptive"_a = remap.adaptive,                    
                    "remap_scaling"_a = remap_scale,
                    "remap_verbose"_a = remap.verbose,
                    "remap_gradient"_a = remap.gradient,
                    //beam
                    "beam"_a = py_beam,
                    "beam_charge"_a = beam.charge,
                    "beam_output"_a = beam.output,  
                    "beam_output_start"_a = beam.dump_start,
                    "beam_output_interval"_a = beam.dump_interval, 
                    // wavelets                                       
                    "wavelet_x"_a = wave_x,
                    "wavelet_y"_a = wave_y,
                    #if DIM == 3
                      "wavelet_z"_a = wave_z,
                    #endif
                    "wavelet_field"_a = wave_field,
                    "wavelet_type"_a = static_cast<int>(wavelets.frame),
                    "use_wavelet_for_subcycle"_a = wavelets.subcycle,
                    "wavelet_output"_a = wavelets.output,
                    "wavelet_output_start"_a = wavelets.dump_start,
                    "wavelet_output_interval"_a = wavelets.dump_interval, 
                    "num_wavelet_files"_a = wavelets.num_wavelet_files,
                    "num_wavelet_fields"_a = wavelets.num_fields,
                    // analysis                         
                    "postprocess"_a = analysis.set,
                    "field_func"_a = analysis.field_expr,
                    "file_exact"_a = analysis.file_exact,
                    "file_error"_a = analysis.file_error,
                    "coord_shift"_a = coord_shift,
                    "coord_scale"_a = coord_scale,
                    "field_scale"_a = analysis.field_scale,
                    // mpi
                    "mpi_rank"_a = mpi.rank,
                    "num_ranks"_a = mpi.num_ranks);

  // parse input file
  py::eval_file(argv[1], py::globals(), locals);

  // store changed python variable values
  // kernel
  kernel.run_name        = locals["run_name"].cast<std::string>();
  kernel.num_wavefronts  = locals["num_wavefronts"].cast<int>();
  kernel.num_dirs        = locals["num_dirs"].cast<int>();
  kernel.num_step        = locals["num_step"].cast<int>();
  kernel.num_particles   = locals["num_particles"].cast<int>();
  kernel.qm              = locals["qm"].cast<double>();  
  kernel.emission_start_step = locals["emission_start"].cast<int>();  
  kernel.emission_interval  = locals["emission_interval"].cast<int>();  
  kernel.print_interval  = locals["print_interval"].cast<int>();
  kernel.min_emit_angle  = locals["min_emit_angle"].cast<double>();
  kernel.dt              = locals["dt"].cast<double>();
  kernel.radius          = locals["radius"].cast<double>();
  params                 = py::cast<py::array>(locals["parameters"]);
  // mesh
  mesh.num_hor           = locals["num_gridpt_hor"].cast<int>();
  mesh.num_ver           = locals["num_gridpt_ver"].cast<int>();
  mesh.span_angle        = locals["mesh_span_angle"].cast<double>();
  mesh.width             = locals["mesh_width"].cast<double>();
  mesh.output            = locals["mesh_output"].cast<bool>();
  mesh.dump_start        = locals["mesh_output_start"].cast<int>();
  mesh.dump_interval     = locals["mesh_output_interval"].cast<int>(); 
  // beam 
  beam.charge            = locals["beam_charge"].cast<double>();
  beam.output            = locals["beam_output"].cast<bool>();  
  beam.dump_start        = locals["beam_output_start"].cast<int>();
  beam.dump_interval     = locals["beam_output_interval"].cast<int>();
  // remap  
  remap.start_step       = locals["remap_start"].cast<int>();
  remap.interval         = locals["remap_interval"].cast<int>();
  remap.scatter          = locals["remap_scatter"].cast<bool>();
  remap.adaptive         = locals["remap_adaptive"].cast<bool>();  
  remap_scale            = py::cast<py::array>(locals["remap_scaling"]);
  remap.verbose          = locals["remap_verbose"].cast<bool>();
  remap.gradient         = locals["remap_gradient"].cast<bool>();
  // wavelets
  wavelets.output        = locals["wavelet_output"].cast<bool>();
  wavelets.dump_start    = locals["wavelet_output_start"].cast<int>();
  wavelets.dump_interval = locals["wavelet_output_interval"].cast<int>();
  wavelets.num_wavelet_files = locals["num_wavelet_files"].cast<int>();
  wavelets.num_fields    = locals["num_wavelet_fields"].cast<int>();

  switch (locals["trajectory_type"].cast<int>()) {
    case 1: kernel.trajectory = STRAIGHT; break;
    case 2: kernel.trajectory = CIRCULAR; break;
    case 3: kernel.trajectory = SINUSOIDAL; break;
    default: throw std::runtime_error("invalid trajectory type");
  }

  auto request_params = static_cast<double*>(params.request().ptr);
  kernel.motion_params[0] = request_params[0];
  kernel.motion_params[1] = request_params[1];

  auto request_scale = static_cast<double*>(remap_scale.request().ptr);
  remap.scaling[0] = request_scale[0];
  remap.scaling[1] = request_scale[1];

  // toggle beam flag
  beam.found = locals.contains("beam");
  if (beam.found) {
    if (mpi.rank == 0) { std::cout << "Initial beam found." << std::endl; }

    py_beam = py::cast<py::array>(locals["beam"]);
    auto beam_request = py_beam.request();
    int const nb = beam_request.shape[0];
    int const dim = beam_request.shape[1];
    beam.count = nb;
    assert(nb == kernel.num_particles);
    beam.particles = static_cast<double*>(beam_request.ptr);

  } else if (mpi.rank == 0) { std::cout << "No prescribed beam." << std::endl; }

  // toggle wavelets flag
  wavelets.found = locals.contains("wavelet_x") and
                   locals.contains("wavelet_y") and
                   locals.contains("wavelet_field");

  if (wavelets.found) {
    wavelets.subcycle = locals["use_wavelet_for_subcycle"].cast<bool>();
    wavelets.count = py::cast<py::array>(locals["wavelet_x"]).request().shape[0];
    wavelets.x     = static_cast<double*>(py::cast<py::array>(locals["wavelet_x"]).request().ptr);
    wavelets.y     = static_cast<double*>(py::cast<py::array>(locals["wavelet_y"]).request().ptr);
    wavelets.field = static_cast<double*>(py::cast<py::array>(locals["wavelet_field"]).request().ptr);

    if (mpi.rank == 0) {
      int imax = (wavelets.count < 10) ? wavelets.count : 10;
      std::cout << "---------------" << std::endl;
      for (int i = 0; i < imax; ++i) {
        std::cout << "wavelet.field["<< i <<"]: "<< wavelets.field[i] << std::endl;
      }
    }  

    if (locals.contains("wavelet_type")) {
      switch (locals["wavelet_type"].cast<int>()) {
        case 0: wavelets.frame = GLOBAL_CARTESIAN; break;
        case 1: wavelets.frame = LOCAL_CARTESIAN; break;
        case 2: wavelets.frame = LOCAL_CYLINDRIC; break;
        default: throw std::runtime_error("invalid wavelets frame");
      }

      if (mpi.rank == 0) {
        std::cout << wavelets.count << " prescribed wavelets with ";
        std::cout << to_string(wavelets.frame) << " found";
        std::cout << (wavelets.subcycle ? " (for subcycle emission only)." : ".");
        std::cout << std::endl;
      }
    }

    analysis.set = locals.contains("postprocess") and locals["postprocess"].cast<bool>();

    if (analysis.set) {
      analysis.gamma = locals["gamma"].cast<double>();
      analysis.shift = static_cast<double*>(py::cast<py::array>(locals["coord_shift"]).request().ptr);
      analysis.scale = static_cast<double*>(py::cast<py::array>(locals["coord_scale"]).request().ptr);
      analysis.field_scale = locals["field_scale"].cast<double>();
      analysis.field_expr = locals["field_func"].cast<std::string>();
      analysis.file_exact = locals["file_exact"].cast<std::string>();
      analysis.file_error = locals["file_error"].cast<std::string>();
    }

  } else if (mpi.rank == 0) { std::cout << "No prescribed wavelets." << std::endl; }

  print();
  timer.stop("process_input_deck");
}

/* -------------------------------------------------------------------------- */
void Input::print() const {

  if (mpi.rank == 0) {
    std::cout << " -----COSYR ver. " << COSYR_VER << "----- "    << std::endl;
    std::cout << "Key simulation parameters:"    << std::endl;
    std::cout << "\u2022 kernel.emission_intervel: " << kernel.emission_interval << std::endl;    
    std::cout << "\u2022 kernel.num_wavefronts: "  << kernel.num_wavefronts << std::endl;
    std::cout << "\u2022 kernel.num_dirs: "        << kernel.num_dirs << std::endl;
    std::cout << "\u2022 kernel.num_step: "        << kernel.num_step << std::endl;
    std::cout << "\u2022 kernel.dt: "              << kernel.dt << std::endl;
    std::cout << "\u2022 kernel.num_particles: "   << kernel.num_particles << std::endl;
    std::cout << "\u2022 kernel.qm: "              << kernel.qm << std::endl;    
    std::cout << "\u2022 kernel.trajectory: "      << kernel.trajectory << std::endl;
    std::cout << "\u2022 kernel.motion_params = [" << kernel.motion_params[0] <<","<< kernel.motion_params[1] << "]" << std::endl;
    if (not beam.found) {
    #if DIM == 3
      std::cout << "\u2022 kernel.pos_elec = ["    << kernel.pos_elec[0] <<","<< kernel.pos_elec[1] <<","<< kernel.pos_elec[2] << "]" << std::endl;
      std::cout << "\u2022 kernel.mom_elec = ["    << kernel.mom_elec[0] <<","<< kernel.mom_elec[1] <<","<< kernel.mom_elec[2] << "]" << std::endl;
    #else
      std::cout << "\u2022 kernel.pos_elec = ["    << kernel.pos_elec[0] <<","<< kernel.pos_elec[1] << "]" << std::endl;
      std::cout << "\u2022 kernel.mom_elec = ["    << kernel.mom_elec[0] <<","<< kernel.mom_elec[1] << "]" << std::endl;
    #endif
    }  
    std::cout << "\u2022 beam.charge: "            << beam.charge << std::endl;
    std::cout << "\u2022 beam.found: "             << beam.found << std::endl;
    if (beam.found) {
      std::cout << "\u2022 beam.count: "           << beam.count << std::endl;
    }
    std::cout << "\u2022 remap.interval: "         << remap.interval << std::endl;
    std::cout << "\u2022 mesh.num_hor: "           << mesh.num_hor << std::endl;
    std::cout << "\u2022 mesh.num_ver: "           << mesh.num_ver << std::endl;
    std::cout << "\u2022 mesh.span_angle: "        << mesh.span_angle << std::endl;
    std::cout << "\u2022 mesh.width: "             << mesh.width << std::endl;
    if (wavelets.found) {
      std::cout << "\u2022 wavelets.frame: "       << to_string(wavelets.frame) << std::endl;
    }
    if (analysis.set) {
      std::cout << "\u2022 analysis.gamma: "       << analysis.gamma << std::endl;
      std::cout << "\u2022 analysis.shift: ["      << analysis.shift[0] <<", "<< analysis.shift[0] << "]" << std::endl;
      std::cout << "\u2022 analysis.scale: ["      << analysis.scale[0] <<", "<< analysis.scale[0] << "]" << std::endl;
      std::cout << "\u2022 analysis.field_scale: " << analysis.field_scale << std::endl;
      std::cout << "\u2022 analysis.field_expr: "  << analysis.field_expr << std::endl;
      std::cout << "\u2022 analysis.file_exact: "  << analysis.file_exact << std::endl;
      std::cout << "\u2022 analysis.file_error: "  << analysis.file_error << std::endl;
    }
    std::cout << " -------------------------- "    << std::endl;
  }
}

/* -------------------------------------------------------------------------- */
void Input::print_step(int i, double t) const {
  if (mpi.rank == 0 and (i % kernel.print_interval == 0)) {
    std::cout << "step " << i << ", t = " << t << std::endl;
  }
}

/* -------------------------------------------------------------------------- */
int Input::finalize() {
  Kokkos::finalize();
  MPI_Finalize();
  return EXIT_SUCCESS;
}

/* -------------------------------------------------------------------------- */
std::string to_string(Frame frame) {
  switch (frame) {
    case LOCAL_CARTESIAN: return "local cartesian coords";
    case LOCAL_CYLINDRIC: return "local cylindrical coords";
    default: return "global cartesian";
  }
}

/* -------------------------------------------------------------------------- */
} // namespace cosyr

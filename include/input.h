#pragma once

// py
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/eval.h>
#include <pybind11/numpy.h>

#include "cosyr.h"

// canonical alias for pybind
namespace py = pybind11;

namespace cosyr {

/**
 * @enum Coordinate system to use for wavelet points:
 *       - global cartesian (x,y)
 *       - local to mesh center (x',y')
 *       - local to mesh center (alpha,chi)
 */
enum Frame : int { GLOBAL_CARTESIAN, LOCAL_CARTESIAN, LOCAL_CYLINDRIC };

/**
 * @enum Electron trajectory type.
 *
 */
enum Traject : int { STRAIGHT=1, CIRCULAR, SINUSOIDAL };

#pragma GCC visibility push(hidden)
/**
 * @class Cache simulation parameters from input deck.
 *
 */
class Input {
public:

  py::dict locals;                   /** input deck parameters */

  struct {
    std::string run_name = "data";   /** name of the simulation run */  
    Traject trajectory = CIRCULAR;   /** trajectory type */
    int emission_start_step = 0;     /** first step to turn on wavelet emission */
    int emission_interval = 1;       /** timestep interval between wavelet move */
    int num_wavefronts = 300;        /** number of wavefronts */
    int num_dirs = 200;              /** number of field lines */
    int num_step = 300;              /** number of steps */
    int num_particles = 2;           /** number of particles */
    double qm = -1.0;                /** charge over mass */ 
    double motion_params[2] = {10.0, 100.0};  /** electron motion parameters */
    double min_emit_angle = 1.0;     /** wavefront minimal emission angle */
    double dt = 0.0001;              /** time step normalized by characteristic frequency */
    double pos_elec[DIM] = {0.0, 1.0}; /** initial position */
    double mom_elec[DIM] = {0.0, 0.0}; /** initial velocity */
    double radius = 1.0;             /** normalized radius */
    int print_interval = 10;         /** timestep interval for printing */
  } kernel;

  struct {
    int start_step = 0;              /** first step to turn on remapping for beam self field */
    int interval = 1;                /** timestep interval between remaps */
    double scaling[2] = {1.0, 1.0};  /** support scaling factor */
    bool adaptive = false;           /** use adaptive smoothing lengths */ 
    bool scatter = false;            /** use scatter weights form */
    bool verbose = false;            /** verbose message about remapping */
    bool gradient = false;          /** whether to compute gradients of remapped fields */
  } remap;

  struct {
    int num_hor = 101;               /** number of horizontal points */
    int num_ver = 101;               /** number of vertical points */
    double width = 0.002;            /** width in unit of radius */
    double span_angle = 0.01;        /** span angle in radians */
    bool output = false;             /** whether to write mesh file */  
    int dump_start = 0;              /** starting step of mesh field dump */
    int dump_interval = 1;           /** time step interval for mesh field dump */            
  } mesh;

  struct {
    bool found = false;              /** whether initial beam provided or not */
    int count = 0;                   /** number of particles */
    double* particles = nullptr;     /** particle beam */
    double charge = 0.1;             /** beam charge in nC **/  
    bool output = false;             /** whether to write beam files */    
    int dump_start = 0;              /** starting step of beam dump */
    int dump_interval = 1;           /** time step interval for beam dump */     
  } beam;

  struct {
    bool subcycle = true;            /** use wavelets for subcycle or not */
    bool found = false;              /** whether provided or not */
    int count = 0;                   /** number of points **/
    double* x = nullptr;             /** coordinates in x */
    double* y = nullptr;             /** coordinates in y */
    double* field = nullptr;         /** wavelets field */
    Frame frame = LOCAL_CARTESIAN;   /** coordinates system */
    bool output = false;             /** whether to write wavelet files */
    int dump_start = 0;              /** starting step of wavelet dump */
    int dump_interval = 1;           /** time step interval for wavelet dump */ 
    int num_wavelet_files = 2;       /** maximum number of wavelet files to output */
    int num_fields = 3;              /** number of fields */
  } wavelets;

  struct {
    bool set = false;                /** analysis parameters set or not */
    double gamma = 1000.;            /** gamma value */
    double* shift = nullptr;         /** coordinates shift terms (x,y)*/
    double* scale = nullptr;         /** coordinates scale factors (x,y)*/
    double field_scale = 1.0;        /** field values scaling */
    std::string field_expr;          /** formula used to compute field */
    std::string file_exact;          /** file path to export exact values */
    std::string file_error;          /** file path to export error map */
  } analysis;

  struct {
    int rank = 0;                    /** current rank */
    int num_ranks = 1;               /** number of ranks */
    MPI_Comm comm = MPI_COMM_NULL;   /** communicator */
  } mpi;

  Timer& timer;                      /** timer */

  /* nb: no wavefront emission between
   * (-min_emit_angle/gamma, min_emit_angle/gamma) */

  /* nb: if subcycle is true, loaded wavelets will be
   * repeatedly emitted at each step and copied into
   * internal wavelets array, otherwise they will only
   * be used for testing interpolation and not
   * copied into internal wavelets array. */

  /**
   * @brief Create an instance and parse input.
   *
   * @param argc: number of program arguments
   * @param argv: list of arguments
   * @param in_timer: timer
   */
  Input(int argc, char* argv[], Timer& in_timer);

  /**
   * @brief Delete instance.
   *
   */
  ~Input() = default;

  /**
   * @brief Print simulation parameters.
   *
   */
  void print() const;

  /**
   * @brief Print current time step info.
   *
   * @param i: current step.
   * @param t: current time.
   */
  void print_step(int i, double t) const;

  /**
   * @brief Finalize runtime.
   *
   */
  int finalize();
};

#pragma GCC visibility pop
  /**
   * @brief Get printable text for given coordinate frame.
   *
   * @param frame: coordinate frame.
   * @return its string representation.
   */
  std::string to_string(Frame frame);
} // namespace csr

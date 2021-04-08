#pragma once

#include <chrono>
#include <map>

namespace cosyr {

class Timer {
public:
  /**
   * @brief Construct and start overall timing.
   *
   */
  Timer() { start("overall"); }

  /**
   * @brief Start profiling for current step.
   *
   * @param step: the current step name.
   */
  void start(std::string const& step) {
    timer[step] = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG
      std::cout << "start " << step << std::endl;
    #endif
  }

  /**
   * @brief Stop profiling for current step.
   *
   * @param step: the current step name.
   */
  void stop(std::string const& step) {
    auto const& tic = timer[step];
    auto const& toc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = toc - tic;
    elapsed[step] += elapsed_seconds.count();
    #ifdef DEBUG
      std::cout << "stop " << step << std::endl;
    #endif
    start(step); // reset
  }

  /**
   * @brief Retrieve overall time and print results.
   *
   * @param print_header: header to print
   * @param rank: current rank.
   */
  void reset(std::string const& print_header, int rank) {
    stop("overall");

    // TODO: other ranks write to file
    if (rank == 0) {
      std::cout << " ---------- " << print_header + " ----------- " << std::endl;
      // print steps timing
      for (auto&& step : elapsed) {
        std::cout << "time ";
        std::cout << step.first <<": ";
        std::cout << step.second << " s" << std::endl;
      }
    }   

    // reinitialize them
    timer.clear();
    elapsed.clear();
  }

private:
  /**
   * @brief timer instances for each step .
   *
   */
  std::map<std::string, std::chrono::high_resolution_clock::time_point> timer {};

  /**
   * @brief elapsed time for each step.
   *
   */
  std::map<std::string, float> elapsed {};
};

}

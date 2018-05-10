#include <common/CudaWarmup.h>
#include "../include/Config.h"
#include "Solution.h"

#include <spdlog/spdlog.h>

// YAML file containing input parameters
static constexpr char CONFIG_FILE_PATH[] = "../config/ps5.yaml";
static constexpr char LOG_FILE_PATH[] = "ps5.log";

std::shared_ptr<spdlog::logger> _logger, _fileLogger;

int main() {
    // Set up loggers
    std::vector<spdlog::sink_ptr> sinks;
    auto colorStdoutSink = std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
    auto fileSink = std::make_shared<spdlog::sinks::simple_file_sink_mt>(LOG_FILE_PATH);
    sinks.push_back(colorStdoutSink);
    sinks.push_back(fileSink);
    _logger = std::make_shared<spdlog::logger>("logger", begin(sinks), end(sinks));
    _fileLogger = std::make_shared<spdlog::logger>("file_logger", fileSink);
    spdlog::register_logger(_logger);
    spdlog::register_logger(_fileLogger);

    common::warmup();
    _fileLogger->info("GPU warmup done");

    Config config(CONFIG_FILE_PATH);

    sol::runProblem1(config);
    sol::runProblem2(config);
    sol::runProblem3(config);
    sol::runProblem4(config);
}
#include "RoutineFunctions.h"
#include <string>
#include "IO.h"

using namespace std::string_literals;

void calculateCRC16WithGPU(std::vector<std::vector<uint8_t>>&& data, std::vector<uint16_t>&& crcs)
{
    std::vector<std::vector<uint8_t>> reflectedData
    {
        reflect(data.at(0)),
        reflect(data.at(1)),
        reflect(data.at(2)),
        reflect(data.at(3))
    };
    bool overflowed{};
    uint16_t finalXORValue{ UINT16_MAX };
    auto progressFileName = "Progress.txt"s;
    try {
        finalXORValue = loadingValueFromFileInHEX<uint16_t>(progressFileName);
        overflowed = finalXORValue == UINT16_MAX ? false : true;
    }
    catch (std::runtime_error ex)
    {
        writeLogEntry("Progress file is not found", LogLevel::INFO);
    }
    std::chrono::steady_clock::time_point start{};
    std::chrono::steady_clock::time_point end{};
    for (; XNOR(finalXORValue < UINT16_MAX, overflowed); finalXORValue++)
    {
        uint16_t completed = overflowed ? finalXORValue : 0;
        showProgress(completed, UINT16_MAX);
        std::cout << " Final XOR value: " << std::noshowbase << std::hex << std::uppercase << finalXORValue << ' ';
        showRemainingExecutionTime(std::move(start), std::move(end), static_cast<uint16_t>(UINT16_MAX - completed));
        start = std::chrono::steady_clock::now();
        eraseFileAndWriteValueInHEX(finalXORValue, progressFileName);
        auto result = bruteForceCRC16WithGPU(finalXORValue, data, reflectedData, crcs);
        if (result.isInitialized())
        {
            std::cout << '\n' << result << '\n';
            addEntryIntoFile(std::move(result), "Results.txt"s);
        }
        if (finalXORValue == UINT16_MAX)
        {
            overflowed = true;
        }
        end = std::chrono::steady_clock::now();
    }
}

void initAndStartCalculating()
{
    std::vector<std::vector<uint8_t>> data
    {
        loadingBytesInHEXFromFile("1.txt"s),
        loadingBytesInHEXFromFile("2.txt"s),
        loadingBytesInHEXFromFile("3.txt"s),
        loadingBytesInHEXFromFile("4.txt"s)
    };
    auto crcs = loadingFileInHEX<uint16_t>("CRC.txt"s);
    calculateCRC16WithGPU(std::move(data), std::move(crcs));
}
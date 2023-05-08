#pragma once

#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include "CRC16.h"

inline bool XNOR(const bool firstCondition, const bool secondCondition)
{
    return firstCondition == secondCondition;
}

extern CRC16 bruteForceCRC16WithGPU(const uint16_t finalXORValue, const std::vector<std::vector<uint8_t>>& data,
    const std::vector<std::vector<uint8_t>>& reflectedData, const std::vector<uint16_t>& crcs);

void calculateCRC16WithGPU(std::vector<std::vector<uint8_t>>&& data, std::vector<uint16_t>&& crcs);

void initAndStartCalculating();

template <typename T>
inline std::vector<T> reflect(const std::vector<T>& values)
{
    std::vector<T> reflected;
    auto bitCountOfType = sizeof(T) * 8;
    for (const auto& value : values)
    {
        T reflectedValue{};
        for (size_t i{}; i < bitCountOfType; i++)
        {
            uint8_t bitValue = value & (1 << i) ? 1 : 0;
            reflectedValue |= bitValue << ((bitCountOfType - 1) - i);
        }
        reflected.emplace_back(reflectedValue);
    }
    return reflected;
}

template <class T>
void showProgress(const T completed, const T total)
{
    auto percent = std::trunc(10000 * (static_cast<float>(completed) / total)) / 100;
    std::cout << '\r' << "Completed: " << std::dec << std::setw(6) << percent << '%';
}

template <class TimePointType, class IntType>
void showRemainingExecutionTime(TimePointType&& start, TimePointType&& end, const IntType remainingOperationNumber)
{
    using namespace std::chrono;
    using namespace std::chrono_literals;
    auto durationInSeconds = duration_cast<seconds>(end - start) * remainingOperationNumber;
    if (durationInSeconds != 0s)
    {
        std::cout << "Remaining time: " << std::dec;
        auto durationInHours = duration_cast<hours>(durationInSeconds);
        if (durationInHours != 0h)
        {
            std::cout << std::setw(6) << durationInHours.count() << "h, ";
            durationInSeconds -= duration_cast<seconds>(durationInHours);
        }
        auto durationInMinutes = duration_cast<minutes>(durationInSeconds);
        if (durationInMinutes != 0min || durationInHours != 0h)
        {
            std::cout << std::setw(2) << durationInMinutes.count() << "min, ";
            durationInSeconds -= duration_cast<seconds>(durationInMinutes);
        }
        if (durationInSeconds != 0s || durationInMinutes != 0min || durationInHours != 0h)
        {
            std::cout << std::setw(2) << durationInSeconds.count() << "s";
        }
    }
}
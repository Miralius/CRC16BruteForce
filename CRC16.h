#pragma once

#include <iostream>
#include "device_launch_parameters.h"

class CRC16
{
private:
    uint16_t polynome{};
    uint16_t initValue{};
    uint16_t finalXORValue{};
    bool inputReflected{};
    bool resultReflected{};
    bool initialized{};
public:
    inline uint16_t getPolynome() const noexcept
    {
        return polynome;
    }
    inline uint16_t getInitValue() const noexcept
    {
        return initValue;
    }
    inline uint16_t getFinalXORValue() const noexcept
    {
        return finalXORValue;
    }
    inline bool isInputReflected() const noexcept
    {
        return inputReflected;
    }
    inline bool isResultReflected() const noexcept
    {
        return resultReflected;
    }
    inline bool isInitialized() const noexcept
    {
        return initialized;
    }

    inline CRC16() = default;

    inline __device__ CRC16(const uint16_t polynome, const uint16_t initValue, const uint16_t finalXORValue,
        const bool inputReflected, const bool resultReflected) noexcept
        : polynome(polynome)
        , initValue(initValue)
        , finalXORValue(finalXORValue)
        , inputReflected(inputReflected)
        , resultReflected(resultReflected)
        , initialized(true)
    {}
};

inline std::ostream& operator<<(std::ostream& out, const CRC16& data)
{
    return out << std::noshowbase << std::hex << std::uppercase
        << "Polynome: " << data.getPolynome() << ' '
        << "Init value: " << data.getInitValue() << ' '
        << "Final XOR value: " << data.getFinalXORValue() << ' ' << std::dec
        << "Input reflected: " << (data.isInputReflected() ? "yes" : "no") << ' '
        << "Result reflected: " << (data.isResultReflected() ? "yes" : "no");
}

#pragma once
#include <iostream>
#include <string>

class CudaDeviceResetException : public std::exception
{
public:
    template <
        class StringType,
        typename = std::enable_if_t
        <!std::is_base_of
        <CudaDeviceResetException,
        std::decay_t<StringType>
        >::value
        >
    >
    inline CudaDeviceResetException(StringType&& exceptionDescription)
        : std::exception(std::forward<StringType>(exceptionDescription).c_str())
    {}
};
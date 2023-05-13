#pragma once
#include <fstream>
#include <string>

class HEXByteSymbol
{
private:
    unsigned char firstSymbol;
    unsigned char secondSymbol;

public:
    inline HEXByteSymbol() = default;
    
    inline HEXByteSymbol(const unsigned char firstSymbol, const unsigned char secondSymbol)
        : firstSymbol(firstSymbol)
        , secondSymbol(secondSymbol)
    {}
    
    inline operator uint8_t()
    {
        using namespace std::string_literals;
        auto stringHEXByte = ""s + firstSymbol + secondSymbol;
        return static_cast<uint8_t>(std::stoul(stringHEXByte, nullptr, 16));
    }
};

inline std::istream& operator>>(std::istream& in, HEXByteSymbol& obj)
{
    unsigned char firstSymbol{};
    unsigned char secondSymbol{};
    in >> firstSymbol >> secondSymbol;
    if (!in)
    {
        return in;
    }
    obj = HEXByteSymbol(firstSymbol, secondSymbol);
    return in;
}

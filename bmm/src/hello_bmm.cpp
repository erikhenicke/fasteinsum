#include "hello_bmm.h"
#include <iostream>

float hello_bmm(int x, float y) {
    std::cout << "Hello, BMM!" << std::endl;
    std::cout << "In hello_bmm the returned value is: "<< x * y << std::endl;
    return x * y;
}
#include <fstream>
#include <string>
#if __has_include(<algorithm>)
#include <algorithm>
#endif


#include "../toojpeg_cpu.h"
#include "../toojpeg_cuda.h"
#include "../utility.h"

const std::string gpu_data_path("gpu_data.txt");
const std::string cpu_data_path("cpu_data.txt");
const std::string generate_command("generate");
const std::string compare_command("compare");


void generate_data()
{
    std::fstream gpu_data(gpu_data_path);
    std::fstream cpu_data(cpu_data_path);
    

}

void compare_data()
{

}


int main(int argc, char const *argv[])
{
    /* 
        args are generate or compare
        generate generates the data files
        compare compares the generated output files, exits if they do not exist
    */
    if (argc == 2)
    {
        if (generate_command.compare(argv[1]) != 0)
        {
            generate_data();
        } 
        else if (compare_command.compare(argv[1]) != 0)
        {
            compare_data();
        } 
        else
        {
            std::cout << "Error: unrecognized command \'" << argv[1] << "\'" << std::endl;
            std::cout << "Please enter \'generate\' or \'compare\'" << std::endl;
            exit(1);
        }
        
    }
    
    return 0;
}

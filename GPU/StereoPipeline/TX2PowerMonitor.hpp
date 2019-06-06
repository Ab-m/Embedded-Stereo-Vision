#include <benchmark/benchmark.h>
#include <sys/types.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include <sstream>
class TX2PowerMonitor
{
private:
    int cntCPU = 0;
    double sumCPU = 0;
    int cntGPU = 0;
    double sumGPU = 0;
    int cntVin = 0;
    double sumVin = 0;
    double power=0;
    int SoC,CPU,GPU,Vin;


public:
    TX2PowerMonitor(benchmark::State& state);
    ~TX2PowerMonitor() ;

        double readUpdates(int source,int &count);
    void measurePower();
    void reportAverage(benchmark::State& state);
};

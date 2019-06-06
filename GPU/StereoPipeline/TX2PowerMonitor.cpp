#include "TX2PowerMonitor.hpp"


    TX2PowerMonitor::TX2PowerMonitor(benchmark::State& state) {
        GPU = open("/sys/devices/3160000.i2c/i2c-0/0-0040/iio:device0/in_power0_input", O_RDONLY | O_NONBLOCK);
        if (GPU < 0) {
            state.SkipWithError("Failed to read SoC power consumption!");
         }
        Vin = open("/sys/devices/3160000.i2c/i2c-0/0-0041/iio:device1/in_power0_input", O_RDONLY | O_NONBLOCK);
        if (Vin < 0) {
            state.SkipWithError("Failed to read Vin power consumption!");
         }
        CPU = open("/sys/devices/3160000.i2c/i2c-0/0-0041/iio:device1/in_power1_input", O_RDONLY | O_NONBLOCK);
        if (CPU < 0) {
            state.SkipWithError("Failed to read CPU power consumption!");
         }
    }
    TX2PowerMonitor::~TX2PowerMonitor() {
        close(CPU);
        close(GPU);
        close(Vin);
        }

        double TX2PowerMonitor::readUpdates(int source,int &count)
    {
        char buffer[32];
        lseek(source, 0, 0);
        int r = read(source, buffer, 32);
        if (r > 0) {
            buffer[r] = 0;
            char *o = NULL;
                count++;
            return strtod(buffer, &o);

        }
        return 0;
    }
    void TX2PowerMonitor::measurePower()
    {

        sumCPU+=readUpdates(CPU,cntCPU);
        sumGPU+=readUpdates(GPU,cntGPU);
        sumVin+=readUpdates(Vin,cntVin);

    }
    void TX2PowerMonitor::reportAverage(benchmark::State& state)
    {

        std::ostringstream reportstream;
        reportstream << "\n Average CPU power Consumed: " << sumCPU/cntCPU << "mW"
                        "\n Average GPU power Consumed: " << sumGPU/cntGPU << "mW"
                        "\n Average Vin power Consumed: " << sumVin/cntVin << "mW";
        std::string report = reportstream.str();
        state.SetLabel(report);
    }


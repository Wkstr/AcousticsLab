#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>

extern "C" void app_main()
{
    while (1)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

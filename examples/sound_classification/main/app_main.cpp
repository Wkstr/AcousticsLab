
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>

static void executor_task(void *)
{
    while (1)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

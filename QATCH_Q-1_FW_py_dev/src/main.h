#include "Wire.h"

// #define SHUTDOWN_TEC_FOR_TEMP_READS

// Add function declarations here for CPP file:
void nv_init(void);
bool detect_hw_revision(void);
void config_hw_revision(byte hw_rev);
void config_hw_ad9851(void);
void ledWrite(int pin, int value);
// byte EEPROM_read_pid(void);
float byte_to_float(byte b);
byte float_to_byte(float f);
bool connect_to_ethernet(void);
float toSmoothFactor(byte avg_samples);
void stopStreaming(void);
bool search_for_dhcp(void);
int checkForTimeSyncUpdates(void);
bool isTimeForSync(void);
unsigned long getNowEpoch(void);
short getSyncState(void);
unsigned long getSystemTime(void);
unsigned long getSystemTime(bool print_status);
void waitForAdcToSettle(void);
void calibrate(long start, long stop, long step);
byte deltaMag(int val1, int val2);
void sendNTPpacket(const char *address);
bool check_and_correct_micros_rollover(void);
void pogo_button_ISR(void);
void pogo_button_pressed(bool init);

void tft_wakeup();
void tft_screensaver();
void tft_splash(bool dp);
void tft_identify(bool identifying);
void tft_idle();
// void tft_testmode();
// void tft_tempbusy();
void tft_cooldown_start();
void tft_cooldown_prepare();
void tft_cooldown();
void tft_progress_show(float pct);
void tft_tempcontrol();
void tft_initialize();
void tft_measure();

void QATCH_setup();
void QATCH_loop();
#include "unity.h"
#include "Arduino.h"
#include "main.cpp"
// #include "MAX31855.h"

// MAX31855 pins
// #define MAX31855_SO 16  // serial out
// #define MAX31855_CS 10  // chip select
// #define MAX31855_CLK 17 // serial clock
// #define MAX31855_WAIT 0 // signal settle delay (in us)

// MAX31855 max31855 = MAX31855(MAX31855_CLK, MAX31855_CS, MAX31855_SO, MAX31855_WAIT);

String test_runner_name;
ushort current_test_number = 0;
ushort text_height = 0;
ushort text_width = 0;

// global test parameters
const float test_set_get_offsets[] = {-6.35, -5, -2.5, 1, -0.05, 0, 0.05, 1, 2.5, 5, 6.35};
#define NUM_OFFSETS sizeof(test_set_get_offsets) / sizeof(test_set_get_offsets[0])

void QATCH_test_code(String test_runner_name = "test_UNKNOWN", bool fillFirst = false)
{
  tft.begin();
  tft.setRotation(TFT_ROTATION);

  if (fillFirst)
    tft.fillScreen(ILI9341_BLACK);

  // calculate fill color to toggle
  uint16_t fillColor = ILI9341_RED;
  if (!sleep_running && tft.readPixel(0, 0) != fillColor)
  {
    // toggle between red and black border with test code message
    for (int i = 0; i <= 3; i++)
    {
      tft.drawRect(i, i, TFT_WIDTH - i - i - 1, TFT_HEIGHT - i - i - 1, fillColor);
    }
    tft.setFontAdafruit();
    tft.setTextColor(fillColor);
    tft.setTextSize(2);
    tft.setCursor(4, 8);
    tft.print(" TEST CODE: ");
    tft.print(test_runner_name);
    //        " TEST CODE      NEED HELP "
  }

  // slow down command handlers to further indicate test code!!!
  // delay(1000);
}

void setUp(void)
{
  // set stuff up here
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);

  current_test_number++;

  tft.begin();
  tft.setRotation(TFT_ROTATION);
  tft.setFontAdafruit();
  tft.setTextSize(1);
  tft.setCursor(4, text_height * (current_test_number + 3));
  tft.setTextColor(ILI9341_LIGHTGREY);
  // tft.print(" ");
  // if (Unity.NumberOfTests >= 10 && current_test_number < 10)
  //   tft.print(" "); // leading space
  // tft.printf("%u/%u", current_test_number, Unity.NumberOfTests);
  tft.print(" ");
  tft.print(Unity.CurrentTestName);
}

void tearDown(void)
{
  // clean stuff up here
  digitalWrite(LED_BUILTIN, LOW);

  tft.begin();
  tft.setRotation(TFT_ROTATION);
  tft.setFontAdafruit();
  tft.setTextSize(1);
  tft.setCursor(tft.width() - text_width, text_height * (current_test_number + 3));

  if (Unity.CurrentTestFailed)
  {
    tft.setTextColor(ILI9341_RED);
    tft.println("[FAILED]");
  }
  else if (Unity.CurrentTestIgnored)
  {
    tft.setTextColor(ILI9341_YELLOW);
    tft.println("[SKIPPED]");
  }
  else
  {
    tft.setTextColor(ILI9341_GREEN);
    tft.println("[PASSED]");
  }
}

/// @brief Test by taking multiple new reads (up to 100)
///        until the reported celsius temperature changes
/// @bug   Requesting an updated read should be unique
/// @bug   Back-to-back reads must be unique samples
/// @bug   MAX31855 library must wait 100ms between new reads
void test_read_celsius_until_changed(void)
{
  // read celsius until changed (up to 10 sec)
  float temp_last = -1;
  bool temp_changed = false;
  for (int i = 0; i < 100; i++)
  {
    digitalToggle(LED_BUILTIN);
    float temp_now = max31855.readCelsius(true);
    TEST_ASSERT_FLOAT_IS_NOT_NAN(temp_now);
    TEST_ASSERT_EQUAL(0, max31855.status());

    if (temp_last == -1)
    {
    }
    else if (temp_last != temp_now)
    {
      temp_changed = true;
      break;
    }
    temp_last = temp_now;
  }
  TEST_ASSERT_TRUE(temp_changed);

  String str_expected = "OK[NONE]";
  char chars_expected[str_expected.length() + 1]; // trailing NULL
  str_expected.toCharArray(chars_expected, sizeof(chars_expected));
  String str_actual = max31855.error();
  char chars_actual[str_actual.length() + 1]; // trailing NULL
  str_actual.toCharArray(chars_actual, sizeof(chars_actual));
  TEST_ASSERT_EQUAL_STRING(&chars_expected, &chars_actual);
}

/// @brief Test by taking multiple new reads (up to 100)
///        until the reported fahrenheit temperature changes
/// @bug   Requesting an updated read should be unique
/// @bug   Back-to-back reads must be unique samples
/// @bug   MAX31855 library must wait 100ms between new reads
void test_read_fahrenheit_until_changed(void)
{
  // read fahrenheit until changed (up to 10 sec)
  float temp_last = -1;
  bool temp_changed = false;
  for (int i = 0; i < 100; i++)
  {
    digitalToggle(LED_BUILTIN);
    float temp_now = max31855.readFahrenheit(true);
    TEST_ASSERT_FLOAT_IS_NOT_NAN(temp_now);
    TEST_ASSERT_EQUAL(0, max31855.status());

    if (temp_last == -1)
    {
    }
    else if (temp_last != temp_now)
    {
      temp_changed = true;
      break;
    }
    temp_last = temp_now;
  }
  TEST_ASSERT_TRUE(temp_changed);

  String str_expected = "OK[NONE]";
  char chars_expected[str_expected.length() + 1]; // trailing NULL
  str_expected.toCharArray(chars_expected, sizeof(chars_expected));
  String str_actual = max31855.error();
  char chars_actual[str_actual.length() + 1]; // trailing NULL
  str_actual.toCharArray(chars_actual, sizeof(chars_actual));
  TEST_ASSERT_EQUAL_STRING(&chars_expected, &chars_actual);
}

/// @brief Test by taking multiple new reads (up to 100)
///        until the reported internal temperature changes
/// @bug   Requesting an updated read should be unique
/// @bug   Back-to-back reads must be unique samples
/// @bug   MAX31855 library must wait 100ms between new reads
void test_read_internal_until_changed(void)
{
  // read internal until changed (up to 10 sec)
  float temp_last = -1;
  bool temp_changed = false;
  for (int i = 0; i < 100; i++)
  {
    digitalToggle(LED_BUILTIN);
    float temp_now = max31855.readInternal(true);
    TEST_ASSERT_FLOAT_IS_NOT_NAN(temp_now);
    TEST_ASSERT_EQUAL(0, max31855.status());

    if (temp_last == -1)
    {
    }
    else if (temp_last != temp_now)
    {
      temp_changed = true;
      break;
    }
    temp_last = temp_now;
  }
  TEST_ASSERT_TRUE(temp_changed);

  String str_expected = "OK[NONE]";
  char chars_expected[str_expected.length() + 1]; // trailing NULL
  str_expected.toCharArray(chars_expected, sizeof(chars_expected));
  String str_actual = max31855.error();
  char chars_actual[str_actual.length() + 1]; // trailing NULL
  str_actual.toCharArray(chars_actual, sizeof(chars_actual));
  TEST_ASSERT_EQUAL_STRING(&chars_expected, &chars_actual);
}

/// @brief Test by taking multiple stale reads (up to 100)
///        checking the reported celsius temperature never changes
/// @bug   Requesting a stale read should be unchanged
/// @bug   Back-to-back stale reads should be the same and quick
/// @bug   MAX31855 library must NOT wait 100ms between stale reads
void test_read_celsius_not_changed(void)
{
  // read celsius not changed (for 100 reads)
  float temp_last = -1;
  bool temp_changed = false;
  for (int i = 0; i < 100; i++)
  {
    digitalToggle(LED_BUILTIN);
    float temp_now = max31855.readCelsius(false);
    TEST_ASSERT_FLOAT_IS_NOT_NAN(temp_now);
    TEST_ASSERT_EQUAL(0, max31855.status());

    if (temp_last == -1)
    {
    }
    else if (temp_last != temp_now)
    {
      temp_changed = true;
      break;
    }
    temp_last = temp_now;
  }
  TEST_ASSERT_FALSE(temp_changed);

  String str_expected = "OK[NONE]";
  char chars_expected[str_expected.length() + 1]; // trailing NULL
  str_expected.toCharArray(chars_expected, sizeof(chars_expected));
  String str_actual = max31855.error();
  char chars_actual[str_actual.length() + 1]; // trailing NULL
  str_actual.toCharArray(chars_actual, sizeof(chars_actual));
  TEST_ASSERT_EQUAL_STRING(&chars_expected, &chars_actual);
}

/// @brief Test by taking multiple new reads (up to 100)
///        checking the reported fahrenheit temperature never changes
/// @bug   Requesting a stale read should be unchanged
/// @bug   Back-to-back stale reads should be the same and quick
/// @bug   MAX31855 library must NOT wait 100ms between stale reads
void test_read_fahrenheit_not_changed(void)
{
  // read fahrenheit not changed (for 100 reads)
  float temp_last = -1;
  bool temp_changed = false;
  for (int i = 0; i < 100; i++)
  {
    digitalToggle(LED_BUILTIN);
    float temp_now = max31855.readFahrenheit(false);
    TEST_ASSERT_FLOAT_IS_NOT_NAN(temp_now);
    TEST_ASSERT_EQUAL(0, max31855.status());

    if (temp_last == -1)
    {
    }
    else if (temp_last != temp_now)
    {
      temp_changed = true;
      break;
    }
    temp_last = temp_now;
  }
  TEST_ASSERT_FALSE(temp_changed);

  String str_expected = "OK[NONE]";
  char chars_expected[str_expected.length() + 1]; // trailing NULL
  str_expected.toCharArray(chars_expected, sizeof(chars_expected));
  String str_actual = max31855.error();
  char chars_actual[str_actual.length() + 1]; // trailing NULL
  str_actual.toCharArray(chars_actual, sizeof(chars_actual));
  TEST_ASSERT_EQUAL_STRING(&chars_expected, &chars_actual);
}

/// @brief Test by taking multiple new reads (up to 100)
///        checking the reported internal temperature never changes
/// @bug   Requesting a stale read should be unchanged
/// @bug   Back-to-back stale reads should be the same and quick
/// @bug   MAX31855 library must NOT wait 100ms between stale reads
void test_read_internal_not_changed(void)
{
  // read internal not changed (for 100 reads)
  float temp_last = -1;
  bool temp_changed = false;
  for (int i = 0; i < 100; i++)
  {
    digitalToggle(LED_BUILTIN);
    float temp_now = max31855.readInternal(false);
    TEST_ASSERT_FLOAT_IS_NOT_NAN(temp_now);
    TEST_ASSERT_EQUAL(0, max31855.status());

    if (temp_last == -1)
    {
    }
    else if (temp_last != temp_now)
    {
      temp_changed = true;
      break;
    }
    temp_last = temp_now;
  }
  TEST_ASSERT_FALSE(temp_changed);

  String str_expected = "OK[NONE]";
  char chars_expected[str_expected.length() + 1]; // trailing NULL
  str_expected.toCharArray(chars_expected, sizeof(chars_expected));
  String str_actual = max31855.error();
  char chars_actual[str_actual.length() + 1]; // trailing NULL
  str_actual.toCharArray(chars_actual, sizeof(chars_actual));
  TEST_ASSERT_EQUAL_STRING(&chars_expected, &chars_actual);
}

void test_set_get_OffsetA(void)
{
  // since OffsetA involves persistent EEPROM,
  // subsequent test must set it back to zero!
  for (uint i = 0; i < NUM_OFFSETS; i++)
  {
    float expected = test_set_get_offsets[i];
    max31855.setOffsetA(expected);
    float actual = max31855.getOffsetA();
    TEST_ASSERT_EQUAL_FLOAT(expected, actual);

    /// @todo EEPROM is no longer set by calls to setOffsetA(), but NVMEM should be tested separately
    // byte sB;
    // if (expected < 0) sB = ~(byte)(-expected * CAL_FACTOR); // don't add 1
    // else sB = (expected * CAL_FACTOR);
    // int old_address = 1; // inverted address mapping
    // int new_address = (EEPROM.length() - 1) - old_address;
    // byte sA = EEPROM.read(new_address);
    // TEST_ASSERT_EQUAL_HEX(sB, sA);
  }
}

void test_set_get_OffsetB(void)
{
  for (uint i = 0; i < NUM_OFFSETS; i++)
  {
    float expected = test_set_get_offsets[i];
    max31855.setOffsetB(expected);
    float actual = max31855.getOffsetB();
    TEST_ASSERT_EQUAL_FLOAT(expected, actual);
  }
}

void test_set_get_OffsetC(void)
{
  for (uint i = 0; i < NUM_OFFSETS; i++)
  {
    float expected = test_set_get_offsets[i];
    max31855.setOffsetC(expected);
    float actual = max31855.getOffsetC();
    TEST_ASSERT_EQUAL_FLOAT(expected, actual);
  }
}

void test_set_get_OffsetH(void)
{
  for (uint i = 0; i < NUM_OFFSETS; i++)
  {
    float expected = test_set_get_offsets[i];
    max31855.setOffsetH(expected);
    float actual = max31855.getOffsetH();
    TEST_ASSERT_EQUAL_FLOAT(expected, actual);
  }
}

void test_clear_offsets(void)
{
  // clear EEPROM from prior OffsetA tests:
  // test_set_get_OffsetA()
  /// @todo Add more OffsetA tests here
  max31855.setOffsetA(0);
  TEST_ASSERT_EQUAL_FLOAT(0, max31855.getOffsetA());

  max31855.setOffsetB(0);
  TEST_ASSERT_EQUAL_FLOAT(0, max31855.getOffsetB());

  max31855.setOffsetC(0);
  TEST_ASSERT_EQUAL_FLOAT(0, max31855.getOffsetC());

  max31855.setOffsetH(0);
  TEST_ASSERT_EQUAL_FLOAT(0, max31855.getOffsetH());

  /// @todo EEPROM is no longer set by calls to setOffsetA(), but NVMEM should be tested separately
  // int old_address = 1; // inverted address mapping
  // int new_address = (EEPROM.length() - 1) - old_address;
  // TEST_ASSERT_EQUAL_HEX(0, EEPROM.read(new_address));
}

void test_float_to_byte_and_back_again(void)
{
  float min_encode = -6.35;
  float max_encode = 6.35;
  float step_size = 0.05;

  // floats have rounding error, and 'expected' really equals 6.350...013 at final step
  int min_x100 = 100 * min_encode;
  int max_x100 = 100 * max_encode;
  int step_x100 = 100 * step_size;
  int trials = 0;

  for (int i = min_x100; i <= max_x100; i += step_x100)
  {
    trials++;
    float expected = i / 100.0;
    Serial.printf("Trial %i: Convert %4.2f\n", trials, expected);
    byte b = float_to_byte(expected);
    float actual = byte_to_float(b);
    TEST_ASSERT_EQUAL(expected, actual);
  }
  TEST_ASSERT_EQUAL(255, trials);
}

int runUnityTests(void)
{
  UNITY_BEGIN();
  RUN_TEST(test_read_celsius_until_changed);
  RUN_TEST(test_read_fahrenheit_until_changed);
  RUN_TEST(test_read_internal_until_changed);
  RUN_TEST(test_read_celsius_not_changed);
  RUN_TEST(test_read_fahrenheit_not_changed);
  RUN_TEST(test_read_internal_not_changed);
  RUN_TEST(test_set_get_OffsetA);
  RUN_TEST(test_set_get_OffsetB);
  RUN_TEST(test_set_get_OffsetC);
  RUN_TEST(test_set_get_OffsetH);
  RUN_TEST(test_clear_offsets);
  RUN_TEST(test_float_to_byte_and_back_again);
  /// @todo Add function test for changing modes H <--> C and reading temp correctly (2 tests: forced updates +/- 1deg and stale reads!)
  /// @todo Create function tests for "icons" library validation
  /// @todo Create function tests for "L298NHB" library validation
  /// @todo Create function tests for "NvMem" library validation
  /// @todo Any other E2E tests that can be run purely in firmware?
  return UNITY_END();
}

/**
 * For Arduino framework
 */
void setup()
{
  // Wait ~2 seconds before the Unity test runner
  // establishes connection with a board Serial interface
  delay(2000);

  // String the_path = __FILE__;
  // int slash_loc = the_path.lastIndexOf('\\');
  // String the_filepath = the_path.substring(0, slash_loc);
  // String the_cpp_name = the_path.substring(slash_loc + 1);
  // int dot_loc = the_cpp_name.lastIndexOf('.');
  // String the_sketchname = the_cpp_name.substring(0, dot_loc);
  // int slash_loc_2 = the_filepath.lastIndexOf('\\');
  // String the_foldername = the_filepath.substring(slash_loc_2 + 1);
  // test_runner_name = the_foldername;

  String the_path = __FILE__;
  String sub_path = the_path.substring(10);
  int slash_loc = sub_path.indexOf('\\');
  test_runner_name = sub_path.substring(0, slash_loc);
  QATCH_test_code(test_runner_name, true);

  tft.setTextSize(1);
  text_height = tft.measureTextHeight((const char *)'X', 1);
  text_width = tft.measureTextWidth((const char *)'X', 1) * 11;

  runUnityTests();

  bool pass = (Unity.TestFailures == 0U);
  tft.begin();
  tft.setRotation(TFT_ROTATION);
  tft.setCursor(4, text_height * (current_test_number + 3));
  tft.setFontAdafruit();
  tft.setTextSize(1);
  tft.setTextColor(ILI9341_LIGHTGREY);
  tft.println();
  tft.println();
  tft.println("  ----------------------- ");
  tft.printf("  %u Tests %u Failures %u Ignored", Unity.NumberOfTests, Unity.TestFailures, Unity.TestIgnores);
  tft.println();
  tft.println();
  tft.setTextColor(pass ? ILI9341_GREEN : ILI9341_RED);
  tft.print("  ");
  tft.setTextSize(2);
  tft.print(pass ? "PASS" : "FAIL");

  delay(1000 * 60 * 15); // 15 mins

  QATCH_setup();
}
void loop()
{
  QATCH_loop();

  QATCH_test_code(test_runner_name);
}

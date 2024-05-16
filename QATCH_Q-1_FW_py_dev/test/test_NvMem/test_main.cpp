#include "unity.h"
#include "Arduino.h"
#include "main.cpp"

/// PERFORM EEPROM WRITES (wears after 100000x cycles)
/// @warning Use this parameter sparingly, not all the time
// #define PERFORM_EEPROM_WRITES

String test_runner_name;
ushort current_test_number = 0;
ushort text_height = 0;
ushort text_width = 0;

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

void test_NvMem_erase(void)
{
#ifndef PERFORM_EEPROM_WRITES
  TEST_IGNORE_MESSAGE("Skip EEPROM writes");
#endif

  nv.erase();

  const int n = nv.size;
  for (int i = 0; i < n; i++)
  {
#if NVMEM_INVERT
    int address = NVMEM_LENGTH - NVMEM_OFFSET - n + i;
#else
    int address = NVMEM_OFFSET + i;
#endif
    TEST_ASSERT_EQUAL(0xFF, EEPROM.read(address));
  }
}

void test_NvMem_load(void)
{
#ifndef PERFORM_EEPROM_WRITES
  bool expect = true;
#else
  bool expect = false;
#endif

  bool valid = nv.load();
  if (!valid)
  {
    Serial.println("EEPROM is empty! Setting defaults.");
    NVMEM = nv.defaults();
    TEST_ASSERT_TRUE(nv.save()); // write required, even if not PERFORM_EEPROM_WRITES
    TEST_ASSERT_TRUE(nv.load());
  }

  TEST_ASSERT_EQUAL(expect, valid);
}

void test_NvMem_default(void)
{
#ifndef PERFORM_EEPROM_WRITES
  TEST_IGNORE_MESSAGE("Skip EEPROM writes");
#endif

  NvMem_RAM DEFAULTS = nv.defaults();
  NVMEM = DEFAULTS;

  TEST_ASSERT_TRUE(nv.save());
  TEST_ASSERT_TRUE(nv.load());

  TEST_ASSERT_EQUAL(DEFAULTS.version, NVMEM.version);
  TEST_ASSERT_EQUAL(DEFAULTS.pid, NVMEM.pid);
  TEST_ASSERT_EQUAL(DEFAULTS.OffsetA, NVMEM.OffsetA);
  /// @note Add new entries here, as extended
}

void test_NvMem_isValid(void)
{
  TEST_ASSERT_TRUE(nv.isValid());
}

void test_NvMem_version(void)
{
  TEST_ASSERT_EQUAL(NVMEM_VERSION, NVMEM.version);
}

void test_NvMem_update(void)
{
/// @note This portion of the test only affects the RAM copy of NVMEM (no EEPROM writes)

/// @attention Update this test as new entries are added to NV struct:
///            In this case, 'OffsetA' is the last entry in NV,
///            and it's a 'byte', so it is set to value '0xFF'
#define LAST_ENTRY NVMEM.OffsetA
#define UNSET_VAL 0xFF

  // prepare input NV struct
  NvMem_RAM DEFAULTS = nv.defaults();
  NVMEM = DEFAULTS;
  NVMEM.version--;   // make NV copy look out-of-date
  NVMEM.pid = 0;     // have a custom value for PID
  LAST_ENTRY = 0xFF; // have an uninit'd value for OffsetA (end of NV struct)

  TEST_ASSERT_TRUE(nv.isValid());

  // assert input NV struct
  TEST_ASSERT_NOT_EQUAL(NVMEM_VERSION, NVMEM.version);
  TEST_ASSERT_NOT_EQUAL(DEFAULTS.version, NVMEM.version);
  TEST_ASSERT_NOT_EQUAL(DEFAULTS.pid, NVMEM.pid);
  TEST_ASSERT_EQUAL(UNSET_VAL, LAST_ENTRY);

  byte changes = nv.update(); // do the update process

  TEST_ASSERT_EQUAL(2, changes); // 'version' and 'OffsetA'
  TEST_ASSERT_TRUE(nv.isValid());

  // assert output NV struct
  TEST_ASSERT_EQUAL(NVMEM_VERSION, NVMEM.version);
  TEST_ASSERT_EQUAL(DEFAULTS.version, NVMEM.version);
  TEST_ASSERT_EQUAL(0, NVMEM.pid); // stays custom value
  TEST_ASSERT_EQUAL(DEFAULTS.OffsetA, NVMEM.OffsetA);
  TEST_ASSERT_NOT_EQUAL(UNSET_VAL, LAST_ENTRY);
}

void test_NvMem_write(void)
{
#ifndef PERFORM_EEPROM_WRITES
  TEST_IGNORE_MESSAGE("Skip EEPROM writes");
#endif

  NvMem_RAM DEFAULTS = nv.defaults();

  // save to EEPROM, and read it back to RAM
  TEST_ASSERT_TRUE(nv.save());
  TEST_ASSERT_TRUE(nv.load());

  byte changes = nv.update(); // do the update process (again)

  TEST_ASSERT_EQUAL(0, changes); // expect no changes
  TEST_ASSERT_TRUE(nv.isValid());

  // assert output NV struct
  TEST_ASSERT_EQUAL(NVMEM_VERSION, NVMEM.version);
  TEST_ASSERT_EQUAL(DEFAULTS.version, NVMEM.version);
  TEST_ASSERT_EQUAL(0, NVMEM.pid); // stays custom value
  TEST_ASSERT_EQUAL(DEFAULTS.OffsetA, NVMEM.OffsetA);
  TEST_ASSERT_NOT_EQUAL(UNSET_VAL, LAST_ENTRY);
}

void test_NvMem_print(void)
{
  nv.load(); // reload from EEPROM

  extern float byte_to_float(byte b); // declared elsewhere
  Serial.println("Loaded NVMEM struct values:");
  Serial.print("version: ");
  Serial.println(NVMEM.version);
  Serial.print("pid: ");
  Serial.println(NVMEM.pid);
  Serial.print("OffsetA: ");
  Serial.print(NVMEM.OffsetA);
  Serial.print(" (");
  Serial.print(byte_to_float(NVMEM.OffsetA));
  Serial.println(")");

  TEST_ASSERT_EQUAL(NVMEM_VERSION, NVMEM.version);
}

void test_NvMem_size(void)
{
  TEST_ASSERT_EQUAL(4, nv.size); // update expected size when adding new entries
}

int runUnityTests(void)
{
  UNITY_BEGIN();
  RUN_TEST(test_NvMem_print);
  RUN_TEST(test_NvMem_size);
  RUN_TEST(test_NvMem_erase); // requires PERFORM_EEPROM_WRITES
  RUN_TEST(test_NvMem_load);  // expectation changes with PERFORM_EEPROM_WRITES
  RUN_TEST(test_NvMem_isValid);
  RUN_TEST(test_NvMem_version);
  RUN_TEST(test_NvMem_update);  // RAM only update checks
  RUN_TEST(test_NvMem_write);   // EEPROM update writes (requires PERFORM_EEPROM_WRITES)
  RUN_TEST(test_NvMem_default); // requires PERFORM_EEPROM_WRITES
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

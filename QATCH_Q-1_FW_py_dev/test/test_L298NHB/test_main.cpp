#include "unity.h"
#include "Arduino.h"
#include "main.cpp"
// #include "L298NHB.h"

// L298NHB pins
// #define L298NHB_M1 1                     // motor signal pin 1 (0=forward,1=back)
// #define L298NHB_E1 2                     // motor enable pin 1 (PWM)
// #define L298NHB_M2 3                     // motor signal pin 2 (0=forward,1=back)
// #define L298NHB_E2 4                     // motor enable pin 2 (PWM)
// #define L298NHB_HEAT 0                   // heat when signal is forward
// #define L298NHB_COOL 1                   // cool when signal is reverse
// #define L298NHB_INIT 1                   // initial PWM enable power
// #define L298NHB_AUTOOFF (1000 * 60 * 15) // time to auto off (in millis)

// L298NHB l298nhb = L298NHB(L298NHB_M1, L298NHB_E1, L298NHB_M2, L298NHB_E2, L298NHB_COOL, L298NHB_INIT);

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

void test_function_should_doBlahAndBlah(void)
{
  // test stuff
  TEST_IGNORE_MESSAGE("Not implemented!");
}

void test_function_should_doAlsoDoBlah(void)
{
  // more test stuff
}

int runUnityTests(void)
{
  UNITY_BEGIN();
  RUN_TEST(test_function_should_doBlahAndBlah);
  RUN_TEST(test_function_should_doAlsoDoBlah);
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

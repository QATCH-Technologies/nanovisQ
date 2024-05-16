#include "unity.h"
#include "Arduino.h"
#include "main.cpp"
// #include "ILI9341_t3.h" // includes <SPI.h>
// #include "font_Poppins-Bold.h"
// #include "icons.h"

// Pins for ILI9341 TFT Touchscreen connections:
// #define TFT_DC 21
// #define TFT_CS 9
// #define TFT_RST 255 // 255 = unused, connect to 3.3V
// #define TFT_MOSI 11
// #define TFT_SCLK 13
// #define TFT_MISO 12

// #define TFT_ROTATION 1

// ILI9341_t3 tft = ILI9341_t3(TFT_CS, TFT_DC, TFT_RST, TFT_MOSI, TFT_SCLK, TFT_MISO);

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

  delay(500);
}

unsigned long testFillScreen()
{
  unsigned long start = micros();
  tft.fillScreen(ILI9341_WHITE);
  delay(250);
  tft.fillScreen(ILI9341_RED);
  delay(250);
  tft.fillScreen(ILI9341_GREEN);
  delay(250);
  tft.fillScreen(ILI9341_BLUE);
  delay(250);
  tft.fillScreen(ILI9341_BLACK);
  delay(250);
  return micros() - start;
}

unsigned long testText()
{
  tft.fillScreen(ILI9341_BLACK);
  unsigned long start = micros();
  tft.setCursor(0, 0);
  tft.setTextColor(ILI9341_WHITE);
  tft.setTextSize(1);
  tft.println("Hello World!");
  tft.setTextColor(ILI9341_YELLOW);
  tft.setTextSize(2);
  tft.println(1234.56);
  tft.setTextColor(ILI9341_RED);
  tft.setTextSize(3);
  tft.println(0xDEADBEEF, HEX);
  tft.println();
  tft.setTextColor(ILI9341_GREEN);
  tft.setTextSize(5);
  tft.println("Groop");
  tft.setTextSize(2);
  tft.println("I implore thee,");
  tft.setTextSize(1);
  tft.println("my foonting turlingdromes.");
  tft.println("And hooptiously drangle me");
  tft.println("with crinkly bindlewurdles,");
  tft.println("Or I will rend thee");
  tft.println("in the gobberwarts");
  tft.println("with my blurglecruncheon,");
  tft.println("see if I don't!");
  return micros() - start;
}

unsigned long testLines(uint16_t color)
{
  unsigned long start, t;
  int x1, y1, x2, y2,
      w = tft.width(),
      h = tft.height();

  tft.fillScreen(ILI9341_BLACK);
  yield();

  x1 = y1 = 0;
  y2 = h - 1;
  start = micros();
  for (x2 = 0; x2 < w; x2 += 6)
    tft.drawLine(x1, y1, x2, y2, color);
  x2 = w - 1;
  for (y2 = 0; y2 < h; y2 += 6)
    tft.drawLine(x1, y1, x2, y2, color);
  t = micros() - start; // fillScreen doesn't count against timing

  yield();
  tft.fillScreen(ILI9341_BLACK);
  yield();

  x1 = w - 1;
  y1 = 0;
  y2 = h - 1;
  start = micros();
  for (x2 = 0; x2 < w; x2 += 6)
    tft.drawLine(x1, y1, x2, y2, color);
  x2 = 0;
  for (y2 = 0; y2 < h; y2 += 6)
    tft.drawLine(x1, y1, x2, y2, color);
  t += micros() - start;

  yield();
  tft.fillScreen(ILI9341_BLACK);
  yield();

  x1 = 0;
  y1 = h - 1;
  y2 = 0;
  start = micros();
  for (x2 = 0; x2 < w; x2 += 6)
    tft.drawLine(x1, y1, x2, y2, color);
  x2 = w - 1;
  for (y2 = 0; y2 < h; y2 += 6)
    tft.drawLine(x1, y1, x2, y2, color);
  t += micros() - start;

  yield();
  tft.fillScreen(ILI9341_BLACK);
  yield();

  x1 = w - 1;
  y1 = h - 1;
  y2 = 0;
  start = micros();
  for (x2 = 0; x2 < w; x2 += 6)
    tft.drawLine(x1, y1, x2, y2, color);
  x2 = 0;
  for (y2 = 0; y2 < h; y2 += 6)
    tft.drawLine(x1, y1, x2, y2, color);

  yield();
  return micros() - start;
}

unsigned long testFastLines(uint16_t color1, uint16_t color2)
{
  unsigned long start;
  int x, y, w = tft.width(), h = tft.height();

  tft.fillScreen(ILI9341_BLACK);
  start = micros();
  for (y = 0; y < h; y += 5)
    tft.drawFastHLine(0, y, w, color1);
  for (x = 0; x < w; x += 5)
    tft.drawFastVLine(x, 0, h, color2);

  return micros() - start;
}

unsigned long testRects(uint16_t color)
{
  unsigned long start;
  int n, i, i2,
      cx = tft.width() / 2,
      cy = tft.height() / 2;

  tft.fillScreen(ILI9341_BLACK);
  n = min(tft.width(), tft.height());
  start = micros();
  for (i = 2; i < n; i += 6)
  {
    i2 = i / 2;
    tft.drawRect(cx - i2, cy - i2, i, i, color);
    yield();
  }

  return micros() - start;
}

unsigned long testFilledRects(uint16_t color1, uint16_t color2)
{
  unsigned long start, t = 0;
  int n, i, i2,
      cx = tft.width() / 2 - 1,
      cy = tft.height() / 2 - 1;

  tft.fillScreen(ILI9341_BLACK);
  n = min(tft.width(), tft.height());
  for (i = n; i > 0; i -= 6)
  {
    i2 = i / 2;
    start = micros();
    tft.fillRect(cx - i2, cy - i2, i, i, color1);
    t += micros() - start;
    // Outlines are not included in timing results
    tft.drawRect(cx - i2, cy - i2, i, i, color2);
    yield();
  }

  return t;
}

unsigned long testFilledCircles(uint8_t radius, uint16_t color)
{
  unsigned long start;
  int x, y, w = tft.width(), h = tft.height(), r2 = radius * 2;

  tft.fillScreen(ILI9341_BLACK);
  start = micros();
  for (x = radius; x < w; x += r2)
  {
    for (y = radius; y < h; y += r2)
    {
      tft.fillCircle(x, y, radius, color);
      yield();
    }
  }

  return micros() - start;
}

unsigned long testCircles(uint8_t radius, uint16_t color)
{
  unsigned long start;
  int x, y, r2 = radius * 2,
            w = tft.width() + radius,
            h = tft.height() + radius;

  // Screen is not cleared for this one -- this is
  // intentional and does not affect the reported time.
  tft.fillScreen(ILI9341_BLACK);
  start = micros();
  for (x = 0; x < w; x += r2)
  {
    for (y = 0; y < h; y += r2)
    {
      tft.drawCircle(x, y, radius, color);
      yield();
    }
  }

  return micros() - start;
}

unsigned long testTriangles()
{
  unsigned long start;
  int n, i, cx = tft.width() / 2 - 1,
            cy = tft.height() / 2 - 1;

  tft.fillScreen(ILI9341_BLACK);
  n = min(cx, cy);
  start = micros();
  for (i = 0; i < n; i += 5)
  {
    tft.drawTriangle(
        cx, cy - i,     // peak
        cx - i, cy + i, // bottom left
        cx + i, cy + i, // bottom right
        tft.color565(i, i, i));
    yield();
  }

  return micros() - start;
}

unsigned long testFilledTriangles()
{
  unsigned long start, t = 0;
  int i, cx = tft.width() / 2 - 1,
         cy = tft.height() / 2 - 1;

  tft.fillScreen(ILI9341_BLACK);
  start = micros();
  for (i = min(cx, cy); i > 10; i -= 5)
  {
    start = micros();
    tft.fillTriangle(cx, cy - i, cx - i, cy + i, cx + i, cy + i,
                     tft.color565(0, i * 10, i * 10));
    t += micros() - start;
    tft.drawTriangle(cx, cy - i, cx - i, cy + i, cx + i, cy + i,
                     tft.color565(i * 10, i * 10, 0));
    yield();
  }

  return t;
}

unsigned long testRoundRects()
{
  unsigned long start;
  int w, i, i2,
      cx = tft.width() / 2 - 1,
      cy = tft.height() / 2 - 1;

  tft.fillScreen(ILI9341_BLACK);
  w = min(tft.width(), tft.height());
  start = micros();
  for (i = 0; i < w; i += 6)
  {
    i2 = i / 2;
    tft.drawRoundRect(cx - i2, cy - i2, i, i, i / 8, tft.color565(i, 0, 0));
    yield();
  }

  return micros() - start;
}

unsigned long testFilledRoundRects()
{
  unsigned long start;
  int i, i2,
      cx = tft.width() / 2 - 1,
      cy = tft.height() / 2 - 1;

  tft.fillScreen(ILI9341_BLACK);
  start = micros();
  for (i = min(tft.width(), tft.height()); i > 20; i -= 6)
  {
    i2 = i / 2;
    tft.fillRoundRect(cx - i2, cy - i2, i, i, i / 8, tft.color565(0, i, 0));
    yield();
  }

  return micros() - start;
}

unsigned long testScreenRotation()
{
  // test sequence: 2, 3, 0, 1
  unsigned long start, duration = 0;
  for (uint8_t rotation = 2; rotation <= 5; rotation++)
  {
    tft.setRotation(rotation % 4);
    start = micros();
    testText();
    duration += (micros() - start);
    delay(1000);
  }
  return duration;
}

unsigned long testReadScreen()
{
  unsigned long start = micros();
  int x = tft.width() / 2,
      y = tft.height() / 2;
  uint16_t setColor, readColor, i;
  uint16_t colors[5] = {ILI9341_WHITE, ILI9341_RED, ILI9341_GREEN, ILI9341_BLUE, ILI9341_BLACK};

  for (i = 0; i < 5; i++)
  {
    setColor = colors[i];
    tft.fillScreen(setColor);
    readColor = tft.readPixel(x, y);
    TEST_ASSERT_EQUAL(setColor, readColor); // LCD SPI read failure
    delay(250);
  }

  return micros() - start;
}

/// @brief Perform a diagnostics test for debug purposes
/// @attention Must use "Verbose Test" to see Serial output
void test_ReadDiagnostics(void)
{
  // read diagnostics (optional but can help debug problems)
  uint8_t x = tft.readcommand8(ILI9341_RDMODE);
  Serial.print("Display Power Mode: 0x");
  Serial.println(x, HEX);
  x = tft.readcommand8(ILI9341_RDMADCTL);
  Serial.print("MADCTL Mode: 0x");
  Serial.println(x, HEX);
  x = tft.readcommand8(ILI9341_RDPIXFMT);
  Serial.print("Pixel Format: 0x");
  Serial.println(x, HEX);
  x = tft.readcommand8(ILI9341_RDIMGFMT);
  Serial.print("Image Format: 0x");
  Serial.println(x, HEX);
  x = tft.readcommand8(ILI9341_RDSELFDIAG);
  Serial.print("Self Diagnostic: 0x");
  Serial.println(x, HEX);
}

void test_FillScreen(void)
{
  unsigned long t = testFillScreen();
  Serial.print(F("Screen fill              "));
  Serial.println(t);
}

void test_Text(void)
{
  unsigned long t = testText();
  Serial.print(F("Text                     "));
  Serial.println(t);
}

void test_Lines(void)
{
  unsigned long t = testLines(ILI9341_CYAN);
  Serial.print(F("Lines                    "));
  Serial.println(t);
}

void test_FastLines(void)
{
  unsigned long t = testFastLines(ILI9341_RED, ILI9341_BLUE);
  Serial.print(F("Horiz/Vert Lines         "));
  Serial.println(t);
}

void test_Rects(void)
{
  unsigned long t = testRects(ILI9341_GREEN);
  Serial.print(F("Rectangles (outline)     "));
  Serial.println(t);
}

void test_FilledRects(void)
{
  unsigned long t = testFilledRects(ILI9341_YELLOW, ILI9341_MAGENTA);
  Serial.print(F("Rectangles (filled)      "));
  Serial.println(t);
}

void test_Circles(void)
{
  unsigned long t = testCircles(10, ILI9341_WHITE);
  Serial.print(F("Circles (outline)        "));
  Serial.println(t);
}

void test_FilledCircles(void)
{
  unsigned long t = testFilledCircles(10, ILI9341_MAGENTA);
  Serial.print(F("Circles (filled)         "));
  Serial.println(t);
}

void test_Triangles(void)
{
  unsigned long t = testTriangles();
  Serial.print(F("Triangles (outline)      "));
  Serial.println(t);
}

void test_FilledTriangles(void)
{
  unsigned long t = testFilledTriangles();
  Serial.print(F("Triangles (filled)       "));
  Serial.println(t);
}

void test_RoundRects(void)
{
  unsigned long t = testRoundRects();
  Serial.print(F("Rounded rects (outline)  "));
  Serial.println(t);
}

void test_FilledRoundRects(void)
{
  unsigned long t = testFilledRoundRects();
  Serial.print(F("Rounded rects (filled)   "));
  Serial.println(t);
}

void test_ScreenRotation(void)
{
  unsigned long t = testScreenRotation();
  Serial.print(F("Screen rotate            "));
  Serial.println(t);
}

void test_ReadScreen(void)
{
  unsigned long t = testReadScreen();
  Serial.print(F("Screen read              "));
  Serial.println(t);
}

int runUnityTests(void)
{
  UNITY_BEGIN();
  Serial.begin(9600);
  Serial.println("ILI9341 Test!");
  RUN_TEST(test_ReadDiagnostics);
  Serial.println(F("Benchmark                Time (microseconds)"));
  RUN_TEST(test_FillScreen);
  RUN_TEST(test_Lines);
  RUN_TEST(test_FastLines);
  RUN_TEST(test_Rects);
  RUN_TEST(test_FilledRects);
  RUN_TEST(test_Circles);
  RUN_TEST(test_FilledCircles);
  RUN_TEST(test_Triangles);
  RUN_TEST(test_FilledTriangles);
  RUN_TEST(test_RoundRects);
  RUN_TEST(test_FilledRoundRects);
  RUN_TEST(test_Text);
  RUN_TEST(test_ScreenRotation);
  RUN_TEST(test_ReadScreen);
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

  QATCH_test_code(test_runner_name, false); // reprint for LCD tests

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

#include <LiquidCrystal_I2C.h>

//LiquidCrystal_I2C lcd(0x27, 16, 2);
const int ledRed = 8;

  // ECG Streaming code

// 1 second = 1,000,000 microseconds.
// 1,000,000 / 250 = 4000 microseconds (4ms) per sample.
const unsigned long SAMPLING_INTERVAL_MICROS = 4000;

// Variable to store the time of the last sample
unsigned long lastSampleTime = 0;

void setup() {
  // Start Serial communication
  Serial.begin(115200);
  lastSampleTime = micros();
  //lcd.init();
 // lcd.backlight();
  pinMode(ledRed, OUTPUT);
  digitalWrite(ledRed, HIGH);
}

void loop() {
  
  unsigned long currentTime = micros();
  
    /*lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("ECG Status:");
    lcd.setCursor(0, 1);
    */
  if (currentTime - lastSampleTime >= SAMPLING_INTERVAL_MICROS) {
    lastSampleTime = currentTime;

    // Read the analog value from the ECG sensor (pin A0)
    int ecgValue = analogRead(A0);

    // Send the value over the serial port
    Serial.println(ecgValue);
  }
}
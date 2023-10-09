/*
  Blink

  Turns an LED on for one second, then off for one second, repeatedly.

  Most Arduinos have an on-board LED you can control. On the UNO, MEGA and ZERO
  it is attached to digital pin 13, on MKR1000 on pin 6. LED_BUILTIN is set to
  the correct LED pin independent of which board is used.
  If you want to know what pin the on-board LED is connected to on your Arduino
  model, check the Technical Specs of your board at:
  https://www.arduino.cc/en/Main/Products

  modified 8 May 2014
  by Scott Fitzgerald
  modified 2 Sep 2016
  by Arturo Guadalupi
  modified 8 Sep 2016
  by Colby Newman

  This example code is in the public domain.

  https://www.arduino.cc/en/Tutorial/BuiltInExamples/Blink
*/
#include <Wire.h>

//def constants
const int LED_RED = 13;
const int LED_YEL = 12;
const int LED_GRN = 11;

//def variables
int x;

// the setup function runs once when you press reset or power the board
void setup() {
// initialize serial comms
  Serial.begin(9600);
  Serial.setTimeout(1);

  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GRN, OUTPUT);
}

// the loop function runs over and over again forever
void loop() {

    if (Serial.available() > 0){
      x = Serial.readString().toInt();
      if (x == 1){
        digitalWrite(LED_RED, HIGH);
        digitalWrite(LED_YEL, LOW);
        digitalWrite(LED_GRN, LOW);
      }
      if (x == 3){
        digitalWrite(LED_RED, LOW);
        digitalWrite(LED_YEL, LOW);
        digitalWrite(LED_GRN, HIGH);
      }
      else if (x == 0) {
        digitalWrite(LED_RED, LOW);
        digitalWrite(LED_YEL, LOW);
        digitalWrite(LED_GRN, LOW);
      }
    }
    
    
}

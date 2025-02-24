#include "command.h"

#define DEBUG 0
#define debug_print(arg) \
  if (DEBUG) Serial.println(arg)

/* ------> ARDUINO NANO <------- */

#define MEASURE_PIN A7
#define SET_PIN 3
#define POWER_PIN 2

long maxVoltage = 10000;
#define calib 1.054 /* SET_VALUE / REAL_VALUE = 3500 / 3320.5 */

bool read_query(String& buf) {
  bool recievedFlag = false;
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n')
      break;
    buf += c;
    recievedFlag = true;
    delay(2);
  }
  buf.trim();
  return recievedFlag;
}

Command parse_command(const String& buf) {
  debug_print(String("Parse command: ") + buf);
  if (buf == "*IDN?")
    return Command::IDN;
  else if (buf == "CONF")
    return Command::CONFIGURE;
  else if (buf == "READ?")
    return Command::READ;
  else if (buf == "SET")
    return Command::SET;
  else if (buf == "POWER")
    return Command::POWER;
  return Command::UNKNOWN;
}

Query parse_query(const String& buf) {
  int sepIdx = buf.indexOf(' ');
  if (sepIdx == -1) {
    auto cmd = parse_command(buf);
    return Query{ cmd, "" };
  } else {
    auto cmd = parse_command(buf.substring(0, sepIdx));
    auto arg = buf.substring(sepIdx + 1);
    return Query{ cmd, arg };
  }
}

void query_handler(const Query& query) {
  long argInt = query.arg.toInt();
  double ratio = 0;
  switch (query.cmd) {
    case Command::IDN:
      Serial.println("BERTAN HIGH VOLTAGE arduino analog control v0.1");
      break;
    case Command::CONFIGURE:
      if (argInt != 0)
        maxVoltage = argInt;
      break;
    case Command::READ:
      {
        uint32_t sum = 0;
        const size_t cnt = 20;
        for (uint8_t i = 0; i < cnt; ++i)
          sum += analogRead(MEASURE_PIN);
        ratio = (double)sum / (double)cnt / 1024.0;
        Serial.println(maxVoltage * ratio / calib);
      }
      break;
    case Command::SET:
      if (argInt <= maxVoltage) {
        ratio = (double)argInt / (double)maxVoltage;
        analogWrite(SET_PIN, round(calib * ratio * 255.0));
      }
      break;
    case Command::POWER:
      if (query.arg == "ON")
        digitalWrite(POWER_PIN, HIGH);
      if (query.arg == "OFF")
        digitalWrite(POWER_PIN, LOW);
      break;
    case Command::UNKNOWN:
      Serial.println("UNKNOWN");
      break;
  }
}

void setup() {
  pinMode(MEASURE_PIN, INPUT);
  pinMode(SET_PIN, OUTPUT);
  pinMode(POWER_PIN, OUTPUT);
  Serial.begin(9600);

  digitalWrite(POWER_PIN, LOW);
}

void loop() {
  String buf = "";
  if (read_query(buf)) {
    Query query = parse_query(buf);
    debug_print(String("Query: ") + String(static_cast<int>(query.cmd)) + " " + query.arg);
    query_handler(query);
  }
}
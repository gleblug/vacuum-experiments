#include "command.h"

#define DEBUG 0
#define debug_print(arg) if (DEBUG) Serial.println(arg)

#define MEASURE_PIN A0
#define SET_PIN 5

long maxVoltage = 10000;

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
  return Command::UNKNOWN;
}

Query parse_query(const String& buf) {
  int sepIdx = buf.indexOf(' ');
  if (sepIdx == -1) {
    auto cmd = parse_command(buf);
    return Query{cmd, ""};
  } else {
    auto cmd = parse_command(buf.substring(0, sepIdx));
    auto arg = buf.substring(sepIdx + 1);
    return Query{cmd, arg};
  }
}

void query_handler(const Query& query) {
  long argInt = query.arg.toInt();
  double ratio;
  switch (query.cmd) {
  case Command::IDN:
    Serial.println("BERTAN HIGH VOLTAGE arduino analog control v0.1");
    break;
  case Command::CONFIGURE:
    if (argInt != 0)
      maxVoltage = argInt;
    break;
  case Command::READ:
    ratio = (double)analogRead(MEASURE_PIN) / 1024.0;
    Serial.println(maxVoltage * ratio * 1.0636);
    break;
  case Command::SET:
    if (argInt <= maxVoltage) {
      ratio = (double)argInt / (double)maxVoltage;
      analogWrite(SET_PIN, round(ratio * 255.0));
    }
    break;
  case Command::UNKNOWN:
    Serial.println("UNKNOWN");
    break;
  }
}

void setup() {
  pinMode(MEASURE_PIN, INPUT);
  pinMode(SET_PIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  String buf = "";
  if (read_query(buf)) {
    Query query = parse_query(buf);
    debug_print(String("Query: ") + String(static_cast<int>(query.cmd)) + " " + query.arg);
    query_handler(query);
  }
}
#pragma once

enum class Command {
  IDN,
  CONFIGURE,
  READ,
  SET,
  POWER,
  UNKNOWN
};

struct Query {
  Command cmd;
  String arg;
};
#pragma once

enum class Command {
  IDN,
  CONFIGURE,
  READ,
  SET,
  UNKNOWN
};

struct Query {
  Command cmd;
  String arg;
};
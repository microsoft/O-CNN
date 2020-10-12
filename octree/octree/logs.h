#ifndef _OCTREE_LOGS_H_
#define _OCTREE_LOGS_H_

#ifdef USE_GLOG

#include <glog/logging.h>

#else

#include <iostream>
#include <sstream>

#include "filenames.h"

class MessageLogger {
 public:
  MessageLogger(const char* file, int line, int severity)
      : severity_(severity) {
    std::string filename = extract_filename(std::string(file));
    stream_ << "[" << filename << ":" << line << "] ";
  }

  ~MessageLogger() {
    stream_ << "\n";
    std::cerr << stream_.str() << std::flush;
    if (severity_ > 0) {
      abort();  // When there is a fatal log, we simply abort.
    }
  }

  // Return the stream associated with the logger object.
  std::stringstream& stream() { return stream_; }

 private:
  std::stringstream stream_;
  int severity_;
};

class LoggerVoidify {
 public:
  LoggerVoidify() {}
  // It has to be an operator with a precedence lower than << but higher than ?:
  void operator&(const std::ostream& s) {}
};

#define CHECK(expression) \
  (expression)            \
      ? (void)0           \
      : LoggerVoidify() & MessageLogger((char*)__FILE__, __LINE__, 1).stream()

#define LOG_IF(expression) \
  !(expression)            \
      ? (void)0            \
      : LoggerVoidify() & MessageLogger((char*)__FILE__, __LINE__, 0).stream()

#endif  // USE_GLOG

#endif  // _OCTREE_LOGS_H_

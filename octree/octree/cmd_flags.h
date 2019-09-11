#ifndef CMD_FLAGS_H_
#define CMD_FLAGS_H_

#include <map>
#include <memory>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace cflags {

using std::string;
using std::shared_ptr;

enum Require { kOptional = 0, kRequired};

class CmdFlags {
 public:
  CmdFlags(const string& name, const string& usage, const Require require)
    : name_(name), usage_(usage), require_(require), set_(false) {}
  virtual const string help() const = 0;
  virtual void set_value(const char* argv) = 0;
  const string& name() const { return name_; }
  const string& usage() const { return usage_; }
  const Require& require() const { return require_; }
  const bool& set() { return set_; }

 protected:
  string name_;
  string usage_;
  Require require_;
  bool set_;
};

template<typename Dtype>
class CmdFlags_Dtype : public CmdFlags {
 public:
  CmdFlags_Dtype(const string& name, const Dtype& value, const string& usage,
      const Require require) : CmdFlags(name, usage, require), value_(value) {}
  void set_value(const Dtype& value) {
    value_ = value; set_ = true;
  }
  const Dtype& value() { return value_; }
  virtual const string help() const {
    std::ostringstream oss;
    bool optional = require_ == kOptional;
    if (optional) { oss << "\t["; } else { oss << "\t "; }
    oss << "--" << name_ << "  <" << usage_ << ">";
    if (optional) oss << "=" << value_ << "]";
    oss << "\n";
    return oss.str();
  }
 protected:
  Dtype value_;
};

class CmdFlags_int : public CmdFlags_Dtype<int> {
 public:
  CmdFlags_int(const string& name, const int& value, const string& usage,
      const Require require) : CmdFlags_Dtype<int>(name, value, usage, require) {}
  virtual void set_value(const char* argv) {
    value_ = atoi(argv); set_ = true;
  }
};

class CmdFlags_float : public CmdFlags_Dtype<float> {
 public:
  CmdFlags_float(const string& name, const float& value, const string& usage,
      const Require require) : CmdFlags_Dtype<float>(name, value, usage, require) {}
  virtual void set_value(const char* argv) {
    value_ = static_cast<float>(atof(argv)); set_ = true;
  }
};

class CmdFlags_bool : public CmdFlags_Dtype<bool> {
 public:
  CmdFlags_bool(const string& name, const bool& value, const string& usage,
      const Require require) : CmdFlags_Dtype<bool>(name, value, usage, require) {}
  virtual void set_value(const char* argv) {
    value_ = atoi(argv) != 0;  set_ = true;
  }
};

class CmdFlags_string : public CmdFlags_Dtype<string> {
 public:
  CmdFlags_string(const string& name, const string& value, const string& usage,
      const Require require) : CmdFlags_Dtype<string>(name, value, usage, require) {}
  virtual void set_value(const char* argv) {
    value_.assign(argv);  set_ = true;
  }
};

class FlagRegistry {
 public:
  typedef std::map<string, shared_ptr<CmdFlags> > FlagMap;

  static FlagMap& Registry() {
    static std::unique_ptr<FlagMap> g_registry_(new FlagMap());
    return *g_registry_;
  }

  static void AddFlag(const string& name, shared_ptr<CmdFlags> flag) {
    FlagMap& flag_map = Registry();
    flag_map[name] = flag;
  }

 private:
  // FlagRegistry should never be instantiated -
  // everything is done with its static variables.
  FlagRegistry() {}
};

class FlagRegisterer {
 public:
  FlagRegisterer(const string& name, shared_ptr<CmdFlags> cmd_flag) {
    FlagRegistry::AddFlag(name, cmd_flag);
  }
};

void PrintHelpInfo(const string& info = "") {
  std::cout << info << std::endl;
  std::vector<string> help_info;
  auto& flag_map = FlagRegistry::Registry();
  for (auto& it : flag_map) {
    help_info.push_back(it.second->help());
  }
  std::sort(help_info.begin(), help_info.end());
  for (auto& it : help_info) {
    std::cout << it;
  }
  std::cout << std::endl;
}

bool ParseCmd(int argc, char *argv[]) {
  // parse
  auto& flag_map = FlagRegistry::Registry();
  for (int i = 1; i < argc; i += 2) {
    if (argv[i][0] == '-' && argv[i][1] == '-') {
      string name(&argv[i][2]);
      auto it = flag_map.find(name);
      if (it != flag_map.end()) {
        if (i + 1 >= argc) {
          std::cout << "The parameter " << argv[i] << " is unset!" << std::endl;
          return false;
        }
        it->second->set_value(argv[i + 1]);
        //} else if (name == "help") {
        //  PrintHelpInfo();
      } else {
        std::cout << "Unknown cmd parameter: " << argv[i] << std::endl;
        return false;
      }
    } else {
      std::cout << "Invalid cmd parameter: " << argv[i] << std::endl;
      return false;
    }
  }

  // check
  for (auto& it : flag_map) {
    CmdFlags* pflag = it.second.get();
    if (pflag->require() == kRequired && pflag->set() == false) {
      std::cout << " The parameter --" << pflag->name()
                << " has to be set!\n";
      return false;
    }
  }
  return true;
}
} // namespace cflags

#define DEFINE_CFLAG_VAR(dtype, name, require, val, usage)                      \
  namespace cflags{                                                             \
    static FlagRegisterer g_registor_##name(#name,                              \
      shared_ptr<CmdFlags>(new CmdFlags_##dtype(#name, val, usage, require)));  \
    const dtype& FLAGS_##name = std::dynamic_pointer_cast<CmdFlags_##dtype>(    \
      FlagRegistry::Registry()[#name])->value();                                \
  }                                                                             \
  using cflags::FLAGS_##name

#define DEFINE_int(name, require, default_val, usage)                          \
  DEFINE_CFLAG_VAR(int, name, require, default_val, usage)

#define DEFINE_float(name, require, default_val, usage)                        \
  DEFINE_CFLAG_VAR(float, name, require, default_val, usage)

#define DEFINE_bool(name, require, default_val, usage)                         \
  DEFINE_CFLAG_VAR(bool, name, require, default_val, usage)

#define DEFINE_string(name, require, default_val, usage)                       \
  DEFINE_CFLAG_VAR(string, name, require, default_val, usage)

#endif // CMD_FLAGS_H_

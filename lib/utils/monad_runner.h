#pragma once

#include <functional>

namespace gpumpc {
template <class T> class MonadRunner {
  public:
    MonadRunner(const T &okay) : okay_(okay), code_(okay) {}
    template <class Func, typename... Args>
    MonadRunner &Run(const Func &func, Args &&... args) {
        if (code_ == okay_) {
            auto t = std::bind(func, std::forward<Args>(args)...);
            code_ = t();
        }
        return *this;
    }
    T code() const { return code_; }

  private:
    const T okay_;
    T code_;
};
} // namespace gpumpc

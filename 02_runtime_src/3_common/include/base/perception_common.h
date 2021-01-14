// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef _BASE_PERCEPTION_COMMON_H_
#define _BASE_PERCEPTION_COMMON_H_

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

typedef struct Anchor {
  float cx;
  float cy;
  float w;
  float h;
  Anchor(float cx, float cy, float w, float h) : cx(cx), cy(cy), w(w), h(h) {}

  friend std::ostream &operator<<(std::ostream &os, const Anchor &anchor) {
    os << "[" << anchor.cx << "," << anchor.cy << "," << anchor.w << ","
       << anchor.h << "]";
    return os;
  }
} Anchor;

/**
 * Bounding box definition
 */
typedef struct Bbox {
  float xmin;
  float ymin;
  float xmax;
  float ymax;

  Bbox() {}

  Bbox(float xmin, float ymin, float xmax, float ymax)
      : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}

  friend std::ostream &operator<<(std::ostream &os, const Bbox &bbox) {
    os << "[" << std::fixed << std::setprecision(6) << bbox.xmin << ","
       << bbox.ymin << "," << bbox.xmax << "," << bbox.ymax << "]";
    return os;
  }

  ~Bbox() {}
} Bbox;

typedef struct Detection {
  int id;
  float score;
  Bbox bbox;
  const char *class_name;
  Detection() {}

  Detection(int id, float score, Bbox bbox)
      : id(id), score(score), bbox(bbox) {}

  Detection(int id, float score, Bbox bbox, const char *class_name)
      : id(id), score(score), bbox(bbox), class_name(class_name) {}

  friend bool operator>(const Detection &lhs, const Detection &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const Detection &det) {
    os << "{"
       << R"("bbox")"
       << ":" << det.bbox << ","
       << R"("score")"
       << ":" << det.score << ","
       << R"("id")"
       << ":" << det.id << "}";
    return os;
  }

  ~Detection() {}
} Detection;

typedef struct Classification {
  int id;
  float score;
  const char *class_name;

  Classification() : class_name(0) {}

  Classification(int id, float score, const char *class_name)
      : id(id), score(score), class_name(class_name) {}

  friend bool operator>(const Classification &lhs, const Classification &rhs) {
    return (lhs.score > rhs.score);
  }

  friend std::ostream &operator<<(std::ostream &os, const Classification &cls) {
    os << "{"
       << R"("score")"
       << ":" << cls.score << ","
       << R"("id")"
       << ":" << cls.id << "}";
    return os;
  }

  ~Classification() {}
} Classification;

struct Perception {
  // Perception data
  std::vector<Detection> det;
  std::vector<Classification> cls;
  std::vector<uint8_t> seg;

  // Perception type
  enum {
    DET = (1 << 0),
    CLS = (1 << 1),
    SEG = (1 << 2),
  } type;

  friend std::ostream &operator<<(std::ostream &os, Perception &perception) {
    os << "[";
    if (perception.type == Perception::DET) {
      auto &detection = perception.det;
      for (int i = 0; i < detection.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << detection[i];
      }

    } else if (perception.type == Perception::CLS) {
      auto &cls = perception.cls;
      for (int i = 0; i < cls.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << cls[i];
      }
    } else if (perception.type == Perception::SEG) {
      auto &seg = perception.seg;
      for (int i = 0; i < seg.size(); i++) {
        if (i != 0) {
          os << ",";
        }
        os << static_cast<int>(seg[i]);
      }
    }
    os << "]";
    return os;
  }
};

#endif  // _BASE_PERCEPTION_COMMON_H_

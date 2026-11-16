// Tests of src/json_writer.hpp on exact output strings.
#include <gtest/gtest.h>

#include <math.h>
#include <stdint.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "json_writer.hpp"

namespace {

using p2vbme::JsonWriter;

TEST(JsonWriter, NestedObjectsAndArraysInWrittenOrder) {
    std::string s;
    JsonWriter w(s);
    w.begin_object();
    w.field("zeta", 1);
    w.field("alpha", "x");
    w.key("list");
    w.begin_array();
    w.value(true);
    w.null();
    w.begin_object();
    w.end_object();
    w.begin_array();
    w.end_array();
    w.end_array();
    w.end_object();
    w.finish();
    EXPECT_EQ(s,
              "{\n  \"zeta\": 1,\n  \"alpha\": \"x\",\n  \"list\": [\n    true,\n    null,\n    {},\n    []\n  ]\n}\n");
}

TEST(JsonWriter, EscapesStrings) {
    std::string s;
    JsonWriter w(s);
    w.value(std::string("q\"b\\s/\n\t\r\b\f\x01\x1f é"));
    EXPECT_EQ(s, "\"q\\\"b\\\\s/\\n\\t\\r\\b\\f\\u0001\\u001f é\"");
}

TEST(JsonWriter, InvalidUtf8BecomesReplacementCharacter) {
    std::string s;
    JsonWriter w(s);
    w.value(std::string("a\xff"
                        "b\xc3\xa9"
                        "c\xe2\x82"
                        "d\xed\xa0\x80"
                        "e\xf0\x9f\x98\x80"));
    // 0xff: invalid; c3 a9 (e acute): kept; e2 82: truncated; ed a0 80: a surrogate; f0 9f 98 80: kept
    EXPECT_EQ(s, "\"a\\ufffdb\xc3\xa9"
                 "c\\ufffd\\ufffdd\\ufffd\\ufffd\\ufffde\xf0\x9f\x98\x80\"");
}

TEST(JsonWriter, NumbersRoundTripShortest) {
    std::string s;
    JsonWriter w(s);
    w.begin_array();
    w.value(0.1);                                     // double 0.1
    w.value(0.1f);                                    // float 0.1: shortest for the float
    w.value(1.0f / 3.0f);                             // 0.33333334
    w.value(1e-6);                                    // 1e-06
    w.value(641.50345);                               //
    w.value(std::numeric_limits<int64_t>::min());     // exact 64-bit integers
    w.value(std::numeric_limits<uint64_t>::max());    //
    w.value((int32_t)-7);                             //
    w.value(std::nan(""));                            // NaN and infinities are null
    w.value(std::numeric_limits<double>::infinity()); //
    w.value(-std::numeric_limits<float>::infinity()); //
    w.end_array();
    w.finish();
    EXPECT_EQ(s, "[\n  0.1,\n  0.1,\n  0.33333334,\n  1e-06,\n  641.50345,\n  -9223372036854775808,\n"
                 "  18446744073709551615,\n  -7,\n  null,\n  null,\n  null\n]\n");
}

TEST(JsonWriter, ArrayFieldAndMisuse) {
    std::string s;
    JsonWriter w(s);
    w.begin_object();
    w.array_field("gains", std::vector<double>{-1.5, 0.0});
    EXPECT_THROW(w.value(1), std::logic_error); // a value without a key inside an object
    EXPECT_THROW(w.end_array(), std::logic_error);
    w.end_object();
    w.finish();
    EXPECT_EQ(s, "{\n  \"gains\": [\n    -1.5,\n    0\n  ]\n}\n");

    std::string t;
    JsonWriter u(t);
    u.begin_object();
    EXPECT_THROW(u.finish(), std::logic_error);
    std::string v;
    JsonWriter x(v);
    x.begin_array();
    EXPECT_THROW(x.key("k"), std::logic_error);
}

} // namespace

// Tests of src/json_writer.hpp on exact output strings.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "json_writer.hpp"

namespace {

using vecme::JsonWriter;

// s written as a JSON string value
std::string written(const std::string &s) {
    std::string out;
    JsonWriter w(out);
    w.value(s);
    return out;
}

// UTF-8 encoding of the code point u, by the standard bit layout (independent of the writer's byte-range checks)
std::string utf8(uint32_t u) {
    std::string s;
    if (u < 0x80) {
        s += (char)u;
    } else if (u < 0x800) {
        s += (char)(0xC0 | u >> 6);
        s += (char)(0x80 | (u & 0x3F));
    } else if (u < 0x10000) {
        s += (char)(0xE0 | u >> 12);
        s += (char)(0x80 | (u >> 6 & 0x3F));
        s += (char)(0x80 | (u & 0x3F));
    } else {
        s += (char)(0xF0 | u >> 18);
        s += (char)(0x80 | (u >> 12 & 0x3F));
        s += (char)(0x80 | (u >> 6 & 0x3F));
        s += (char)(0x80 | (u & 0x3F));
    }
    return s;
}

// ---- Structure: objects, arrays, layout ----

// Objects and arrays nested in each other: keys in written order, commas, indentation, empty {} and []
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

// array_field(): a key followed by the vector's elements as an array
TEST(JsonWriter, ArrayField) {
    std::string s;
    JsonWriter w(s);
    w.begin_object();
    w.array_field("gains", std::vector<double>{-1.5, 0.0});
    w.end_object();
    w.finish();
    EXPECT_EQ(s, "{\n  \"gains\": [\n    -1.5,\n    0\n  ]\n}\n");
}

// ---- Numbers ----

// Numbers: shortest decimal that reads back the same, exact 64-bit integers, NaN and infinities as null
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

// ---- Strings: escaping and UTF-8 ----

// Escaping: quote, backslash and control characters escaped; / and non-ASCII characters kept as is
TEST(JsonWriter, EscapesStrings) {
    std::string s;
    JsonWriter w(s);
    w.value(std::string("q\"b\\s/\n\t\r\b\f\x01\x1f é"));
    EXPECT_EQ(s, "\"q\\\"b\\\\s/\\n\\t\\r\\b\\f\\u0001\\u001f é\"");
}

// Valid UTF-8 is kept: every code point from U+0080 to U+10FFFF except the surrogates (U+D800 to U+DFFF)
TEST(JsonWriter, EveryValidUtf8CharacterIsKept) {
    for (uint32_t u = 0x80; u <= 0x10FFFF; ++u) {
        if (u >= 0xD800 && u <= 0xDFFF)
            continue;
        const std::string s = utf8(u);
        ASSERT_EQ(written(s), "\"" + s + "\"") << "U+" << std::hex << u;
    }
}

// Valid and invalid UTF-8 mixed in one string: each invalid piece becomes U+FFFD, the valid ones are kept
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

// Each rule of UTF-8 broken at its boundary: every byte of the invalid sequence becomes U+FFFD
TEST(JsonWriter, InvalidUtf8SequencesAreReplaced) {
    const std::pair<std::string, std::string> cases[] = {
        {"\xc0\x80", "\\ufffd\\ufffd"},                       // overlong (U+0000 in 2 bytes)
        {"\xc1\xbf", "\\ufffd\\ufffd"},                       // overlong (U+007F in 2 bytes)
        {"\xe0\x9f\xbf", "\\ufffd\\ufffd\\ufffd"},            // overlong (U+07FF in 3 bytes)
        {"\xf0\x8f\xbf\xbf", "\\ufffd\\ufffd\\ufffd\\ufffd"}, // overlong (U+FFFF in 4 bytes)
        {"\xed\xa0\x80", "\\ufffd\\ufffd\\ufffd"},            // surrogate U+D800
        {"\xed\xbf\xbf", "\\ufffd\\ufffd\\ufffd"},            // surrogate U+DFFF
        {"\xf4\x90\x80\x80", "\\ufffd\\ufffd\\ufffd\\ufffd"}, // U+110000, beyond U+10FFFF
        {"\xe2\x82", "\\ufffd\\ufffd"},                       // cut off by the end of the string
        {"\xf0\x9f\x98", "\\ufffd\\ufffd\\ufffd"},            // cut off by the end of the string
        {"\xc3\x41", "\\ufffdA"},                             // a first byte followed by ASCII (A)
        {"\xe2\x28\xa1", "\\ufffd(\\ufffd"},                  // second byte not a continuation byte
    };
    for (const auto &[in, out] : cases)
        EXPECT_EQ(written(in), "\"" + out + "\"") << "input " << ::testing::PrintToString(in);
}

// Every byte that can never start a character (80-C1, F5-FF) becomes U+FFFD
TEST(JsonWriter, BytesThatNeverStartACharacterAreReplaced) {
    // each followed by three continuation bytes, which become U+FFFD too
    for (int b = 0x80; b <= 0xFF; ++b) {
        if (b >= 0xC2 && b <= 0xF4)
            continue;
        for (const std::string tail : {"\x80\x80\x80", "\xbf\xbf\xbf"})
            ASSERT_EQ(written((char)b + tail), "\"\\ufffd\\ufffd\\ufffd\\ufffd\"") << "first byte " << std::hex << b;
    }
}

// ---- Misuse: wrong calls throw ----

// Wrong calls throw and leave the output as it was
TEST(JsonWriter, MisuseThrowsAndWritesNothing) {
    std::string s;
    JsonWriter w(s);
    w.begin_object();
    w.array_field("gains", std::vector<double>{-1.5, 0.0});
    EXPECT_THROW(w.value(1), std::logic_error);    // a value without a key inside an object
    EXPECT_THROW(w.end_array(), std::logic_error); // closes an array while an object is open
    w.end_object();
    w.finish();
    EXPECT_EQ(s, "{\n  \"gains\": [\n    -1.5,\n    0\n  ]\n}\n");

    std::string t;
    JsonWriter u(t);
    u.begin_object();
    EXPECT_THROW(u.finish(), std::logic_error); // an object is still open
    std::string v;
    JsonWriter x(v);
    x.begin_array();
    EXPECT_THROW(x.key("k"), std::logic_error); // a key inside an array

    // a key waiting for its value: another key, or closing the object, throws
    std::string a;
    JsonWriter wa(a);
    wa.begin_object();
    wa.key("k");
    EXPECT_THROW(wa.key("other"), std::logic_error); // a second key before the first has its value
    EXPECT_THROW(wa.end_object(), std::logic_error); // closes the object while "k" waits for its value
    wa.value(1);
    wa.end_object();
    wa.finish();
    EXPECT_EQ(a, "{\n  \"k\": 1\n}\n");

    // top level: no key, and only one value
    std::string b;
    JsonWriter wb(b);
    EXPECT_THROW(wb.key("k"), std::logic_error); // a key outside any object
    wb.value(1);
    EXPECT_THROW(wb.value(2), std::logic_error); // a second top-level value
    EXPECT_EQ(b, "1");
}

} // namespace

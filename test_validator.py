"""
Validator unit tests ? Phase 2.3 / 2.5 fixes

Covers:
  - All pre-existing valid formats (must still pass)
  - New district_letter format  (DL3CAY9324 style)
  - New CH state code           (Chandigarh)
  - Invalid inputs that must stay rejected
  - State-code validation
  - Quality-scorer smoke test
"""

import sys
import traceback
from indian_plate_validator import (
    PlateValidator,
    IndianStateCodeValidator,
    IndianPlatePatterns,
    PlateQualityScorer,
)


# ?? helpers ??????????????????????????????????????????????????????????????????

PASS = "PASS"
FAIL = "FAIL"

results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, label, detail))
    icon = "OK" if condition else "FAIL"
    print(f"  [{icon}] {label}" + (f"  ({detail})" if detail else ""))
    return condition


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def v(text):
    """Shorthand: return validator result dict for text."""
    return PlateValidator.validate_indian_plate(text)


# ?? SECTION 1: previously valid formats must still pass ??????????????????????

section("SECTION 1 ? Pre-existing valid formats (must stay valid)")

r = v("MH12AB1234")
check("MH12AB1234 ? valid",           r["valid"])
check("MH12AB1234 ? VALID status",    r["validation_status"] == "VALID")
check("MH12AB1234 ? state=MH",        r["state_code"] == "MH")
check("MH12AB1234 ? pattern=standard",r["pattern_name"] == "standard")

r = v("DL01CD5678")
check("DL01CD5678 ? valid",           r["valid"])
check("DL01CD5678 ? state=DL",        r["state_code"] == "DL")

r = v("KL10AW2814")
check("KL10AW2814 ? valid",           r["valid"])
check("KL10AW2814 ? state=KL",        r["state_code"] == "KL")

r = v("MH12A1234")
check("MH12A1234 ? valid (old_format)",   r["valid"])
check("MH12A1234 ? pattern=old_format",   r["pattern_name"] in ("old_format", "commercial"))

r = v("MH12ABC1234")
check("MH12ABC1234 ? valid (three_wheeler)", r["valid"])


# ?? SECTION 2: new district_letter format ?????????????????????????????????????

section("SECTION 2 ? New district_letter format  (AA#XXX####)")

r = v("DL3CAY9324")
check("DL3CAY9324 ? valid",                r["valid"],
      f"status={r['validation_status']} pattern={r['pattern_name']}")
check("DL3CAY9324 ? VALID status",         r["validation_status"] == "VALID")
check("DL3CAY9324 ? state=DL",             r["state_code"] == "DL")
check("DL3CAY9324 ? pattern=district_letter", r["pattern_name"] == "district_letter")

r = v("MH4ABX1234")
check("MH4ABX1234 ? valid (district_letter)", r["valid"],
      f"status={r['validation_status']}")
check("MH4ABX1234 ? state=MH",            r["state_code"] == "MH")

r = v("UP7CAY9999")
check("UP7CAY9999 ? valid (district_letter)", r["valid"])
check("UP7CAY9999 ? state=UP",            r["state_code"] == "UP")

r = v("KA9ZAB1234")
check("KA9ZAB1234 ? valid (district_letter)", r["valid"])

# boundary: district_letter with invalid state must still be rejected
r = v("XX3CAY9324")
check("XX3CAY9324 ? invalid state code rejected", not r["valid"],
      f"status={r['validation_status']}")
check("XX3CAY9324 ? INVALID_STATE_CODE",  r["validation_status"] == "INVALID_STATE_CODE")


# ?? SECTION 3: CH (Chandigarh) state code ????????????????????????????????????

section("SECTION 3 ? CH (Chandigarh) state code")

# State-code lookup
check("CH in STATE_CODES",
      IndianStateCodeValidator.is_valid_state_code("CH"))
check("CH state name = Chandigarh",
      IndianStateCodeValidator.get_state_name("CH") == "Chandigarh")

# Plate with CH + standard format
r = v("CH01AZ2641")
check("CH01AZ2641 ? valid",           r["valid"],
      f"status={r['validation_status']} pattern={r['pattern_name']}")
check("CH01AZ2641 ? state=CH",        r["state_code"] == "CH")

# Plate with CH + district_letter format
r = v("CH4CAZ1234")
check("CH4CAZ1234 ? valid (CH district_letter)", r["valid"],
      f"status={r['validation_status']}")

# Old format with CH
r = v("CH01A1234")
check("CH01A1234 ? valid (CH old_format)",  r["valid"],
      f"status={r['validation_status']}")

# CH present in error message listing if invalid state
r = v("ZZ12AB1234")
check("Invalid state ZZ ? INVALID_STATE_CODE", r["validation_status"] == "INVALID_STATE_CODE")
check("Error message contains 'CH'", "CH" in r["reason"])


# ?? SECTION 4: invalid inputs must remain rejected ???????????????????????????

section("SECTION 4 ? Invalid inputs (must stay rejected)")

cases_invalid = [
    ("",             "EMPTY",              "empty string"),
    ("   ",          "EMPTY",              "whitespace only"),
    ("12AB1234",     "INVALID_FORMAT",     "starts with digits"),
    ("ABCD1234",     "INVALID_STATE_CODE", "4-letter prefix"),
    ("ZZ12AB1234",   "INVALID_STATE_CODE", "ZZ not a state"),
    ("XY01AB1234",   "INVALID_STATE_CODE", "XY not a state"),
    ("MH12",         "INVALID_FORMAT",     "too short"),
    ("MH1234567890", "INVALID_FORMAT",     "too long / no pattern match"),
    ("H47Y1124",     "INVALID_FORMAT",     "starts with H + digit"),
    ("6891H10N9",    "INVALID_FORMAT",     "starts with digit"),
]

for text, expected_status, note in cases_invalid:
    r = v(text)
    check(
        f"'{text}' ? {expected_status}  [{note}]",
        r["validation_status"] == expected_status and not r["valid"],
        f"got status={r['validation_status']} valid={r['valid']}"
    )


# ?? SECTION 5: state-code validator directly ?????????????????????????????????

section("SECTION 5 ? State-code validator")

valid_codes = ["MH", "DL", "KL", "CH", "AP", "GJ", "KA", "TN", "UP", "WB",
               "HR", "PB", "RJ", "OR", "OD", "JK", "HP", "BR", "AS", "NL"]
for code in valid_codes:
    check(f"is_valid_state_code('{code}')", IndianStateCodeValidator.is_valid_state_code(code))

invalid_codes = ["XX", "YY", "ZZ", "AB", "CD", "EF", "00", "1A", "CH1"]
for code in invalid_codes:
    check(f"is_valid_state_code('{code}') ? False",
          not IndianStateCodeValidator.is_valid_state_code(code))


# ?? SECTION 6: pattern regex sanity ??????????????????????????????????????????

section("SECTION 6 ? Pattern regex sanity (existing + new)")

import re
pattern_tests = [
    # (pattern_name,      text,            should_match)
    ("standard",         "MH12AB1234",    True),
    ("standard",         "DL01CD5678",    True),
    ("standard",         "DL3CAY9324",    False),   # district_letter, not standard
    ("old_format",       "MH12A1234",     True),
    ("three_wheeler",    "MH12ABC1234",   True),
    ("district_letter",  "DL3CAY9324",    True),
    ("district_letter",  "MH12AB1234",    False),   # standard, not district_letter
    ("district_letter",  "CH4CAZ1234",    True),
    ("district_letter",  "UP7CAY9999",    True),
    ("district_letter",  "DL33AY9324",    False),   # 2 digits, that's standard
]

patterns = IndianPlatePatterns.get_all_patterns()
for pname, text, expect in pattern_tests:
    pat = patterns[pname]["pattern"]
    matched = bool(re.match(pat, text))
    check(f"pattern '{pname}' vs '{text}' ? {expect}", matched == expect,
          f"regex={pat}")


# ?? SECTION 7: quality scorer smoke test ?????????????????????????????????????

section("SECTION 7 ? Quality scorer smoke test")

val_valid   = v("MH12AB1234")
val_invalid = v("6891H10N9")

qs1 = PlateQualityScorer.score_plate_quality("MH12AB1234", 0.95, val_valid, 0, 1.0)
check("Valid plate quality_tier ? GOOD",
      qs1["quality_tier"] in ("EXCELLENT", "GOOD"),
      f"tier={qs1['quality_tier']} score={qs1['total_score']:.3f}")
check("Valid plate total_score > 0.7", qs1["total_score"] > 0.7)

qs2 = PlateQualityScorer.score_plate_quality("6891H10N9", 0.74, val_invalid, 0, 1.0)
check("Garbled plate quality_tier ? GOOD",
      qs2["quality_tier"] in ("FAIR", "POOR", "VERY_POOR", "GOOD"),
      f"tier={qs2['quality_tier']} score={qs2['total_score']:.3f}")


# ?? Summary ???????????????????????????????????????????????????????????????????

section("SUMMARY")

total  = len(results)
passed = sum(1 for s, _, _ in results if s == PASS)
failed = total - passed

print(f"\n  Total  : {total}")
print(f"  Passed : {passed}")
print(f"  Failed : {failed}")

if failed:
    print("\n  FAILED tests:")
    for status, label, detail in results:
        if status == FAIL:
            print(f"    ? {label}" + (f"  ({detail})" if detail else ""))

print()
sys.exit(0 if failed == 0 else 1)

#!/usr/bin/env bash

FILENAME="tests/custom-test1.in"

tickers=("GOOG" "APPL" "NVDA")
actions=("B"  "S")

cat << EOF > "$FILENAME"
1
o
EOF

# echo "B 1 GOOG 10 1" >> "$FILENAME"

# echo "B 2 GOOG 10 1" >> "$FILENAME"

# for i in {1..20}; do
#   echo "B $i GOOG 1000 1" >> "$FILENAME"
# done

# for i in {21..40}; do
#   echo "S $i GOOG 1000 1" >> "$FILENAME"
# done

# for i in {41..60}; do
#   echo "B $i GOOG 1000 1" >> "$FILENAME"
# done

# for i in {3..100}; do
#   echo "B $i GOOG 1000 1" >> "$FILENAME"
# done

# for i in {101..198}; do
#   echo "S $i GOOG 1000 1" >> "$FILENAME"
# done

# echo "S 199 GOOG 10 1" >> "$FILENAME"

# echo "S 200 GOOG 10 1" >> "$FILENAME"

# for i in {100002..100004}; do
#   echo "S $i GOOG 1000 1" >> "$FILENAME"
# done

for i in {1..100000}; do
  echo "${actions[$((RANDOM % 2))]} $i ${tickers[$((RANDOM % 3))]} $((RANDOM + 200)) 1" >> "$FILENAME"
done

echo "x" >> "$FILENAME"
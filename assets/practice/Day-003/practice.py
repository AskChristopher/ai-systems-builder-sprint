"""Day 001 practice."""


def main() -> None:
    print("Day 001 practice placeholder")
# Exercise 1 - Trigger five exceptions intentionally.

# 10 / 0 - ZeroDivisionError
# print(x) - NameError
# int("hello") - ValueError
# [1,2][10] - IndexError
# {"name":"Chris"}["age"] - Keyerror

# Exercise 2 - Handle the exceptions using try/except blocks.
"""
try:
    print(10 / 0)
except ZeroDivisionError:
    print("Can't divide by zero.")
"""

"""try:
    number = int(input("Enter a number: "))
except ValueError:
    print("Please enter digits only.")
    """
"""try:
    number = int("hello")
except ValueError as error:
    print(error)
    """


if __name__ == "__main__":
    main()

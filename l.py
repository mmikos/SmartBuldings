from satisfaction_score_data_generator import HI_calculation
import math

heat_index = []
a: float = 17.27
b: float = 237.3


alfa = ((a * 22.23) / (b + 22.23)) + math.log(
    46.77 / 100)

D = (b * alfa) / (a - alfa)

HI = 22.23 - 1.0799 * math.exp(0.03755 * 22.23) * (
        1 - math.exp(0.0801 * (D - 14)))



HI = HI_calculation([[22.01802326], [22.16], [22.23]], [[46.31395349], [51.23], [46.77]], 3)

print("whatever")

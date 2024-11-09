import numpy as np
import typing as tp

INF = int(1e9)

class TransportationProblem:
    def __init__(
            self,
            n_sources: int,
            n_destinations: int,
            supply: tp.List[int],
            demand: tp.List[int],
            cost: tp.List[tp.List[int]]
    ):
        self._n_sources = n_sources
        self._n_destinations = n_destinations
        self._supply = supply
        self._demand = demand
        self._cost = cost

    def find_basic_north_west(self) -> tp.List[tp.Tuple[int, int, int]]:
        supply = self._supply.copy()
        demand = self._demand.copy()
        ans: tp.List[tp.Tuple[int, int, int]] = []

        i, j = 0, 0
        while i < self._n_sources and j < self._n_destinations:
            spent = min(supply[i], demand[j])
            ans.append((i + 1, j + 1, spent))
            supply[i] -= spent
            demand[j] -= spent
            if demand[j] == 0:
                j += 1
            if supply[i] == 0:
                i += 1

        return ans

    def find_basic_vogel(self) -> tp.List[tp.Tuple[int, int, int]]:
        supply = np.array(self._supply)
        demand = np.array(self._demand)
        ans: tp.List[tp.Tuple[int, int, int]] = []

        while np.any(supply) and np.any(demand):
            row_penalty = []
            for i in range(self._n_sources):
                if supply[i] == 0:
                    row_penalty.append(-INF)
                    continue
                row_costs = [self._cost[i][j] for j in range(self._n_destinations) if demand[j] > 0]
                row_penalty.append(0 if len(row_costs) < 2 else sorted(row_costs)[1] - sorted(row_costs)[0])

            col_penalty = []
            for j in range(self._n_destinations):
                if demand[j] == 0:
                    col_penalty.append(-INF)
                    continue
                col_costs = [self._cost[i][j] for i in range(self._n_sources) if supply[i] > 0]
                col_penalty.append(0 if len(col_costs) < 2 else sorted(col_costs)[1] - sorted(col_costs)[0])

            if max(row_penalty) >= max(col_penalty):
                i = row_penalty.index(max(row_penalty))
                j = int(np.argmin([self._cost[i][_] if demand[_] > 0 else float('inf') for _ in range(self._n_destinations)]))
            else:
                j = col_penalty.index(max(col_penalty))
                i = int(np.argmin([self._cost[_][j] if supply[_] > 0 else float('inf') for _ in range(self._n_sources)]))

            spent = min(supply[i], demand[j])
            ans.append((i + 1, j + 1, spent))
            supply[i] -= spent
            demand[j] -= spent

        return ans

    def find_basic_russel(self) -> tp.List[tp.Tuple[int, int, int]]:
        supply = np.array(self._supply)
        demand = np.array(self._demand)
        ans: tp.List[tp.Tuple[int, int, int]] = []

        while np.any(supply) and np.any(demand):
            row_u = []
            for i in range(self._n_sources):
                if supply[i] == 0:
                    row_u.append(-INF)
                    continue
                row_costs = [self._cost[i][j] for j in range(self._n_destinations) if demand[j] > 0]
                row_u.append(max(row_costs))

            column_v = []
            for j in range(self._n_destinations):
                if demand[j] == 0:
                    column_v.append(-INF)
                    continue
                col_costs = [self._cost[i][j] for i in range(self._n_sources) if supply[i] > 0]
                column_v.append(max(col_costs))

            delta_min = INF
            basic_i, basic_j = None, None

            for i in range(self._n_sources):
                if supply[i] == 0:
                    continue
                for j in range(self._n_destinations):
                    if demand[j] == 0:
                        continue
                    delta_ij = self._cost[i][j] - row_u[i] - column_v[j]
                    if delta_ij < delta_min:
                        delta_min = delta_ij
                        basic_i, basic_j = i, j

            spent = min(supply[basic_i], demand[basic_j])
            ans.append((basic_i + 1, basic_j + 1, spent))
            supply[basic_i] -= spent
            demand[basic_j] -= spent

        return ans

# Проверка валидности решения
def is_solution_valid(solution: tp.List[tp.Tuple[int, int, int]], supply: tp.List[int], demand: tp.List[int]) -> bool:
    total_supply_used = sum([x[2] for x in solution])
    total_supply = sum(supply)
    total_demand = sum(demand)

    # Проверяем, что общая использованная поставка равна общей потребности
    if total_supply_used != total_supply or total_supply_used != total_demand:
        return False

    # Проверка полного удовлетворения поставок и спросов
    used_supply = [0] * len(supply)
    used_demand = [0] * len(demand)
    for i, j, amount in solution:
        used_supply[i - 1] += amount
        used_demand[j - 1] += amount

    return used_supply == supply and used_demand == demand

# Test cases
def test_cases():
    test_data = [
        (3, 4, [20, 30, 25], [10, 25, 15, 25], [
            [8, 6, 10, 9],
            [9, 12, 13, 7],
            [14, 9, 16, 5]
        ]),
        (3, 5, [140, 180, 160], [60, 70, 120, 130, 100], [
            [2, 3, 4, 2, 4],
            [8, 4, 1, 4, 1],
            [9, 7, 3, 7, 2]
        ]),
        (4, 5, [50, 60, 50, 50], [30, 20, 70, 30, 60], [
            [16, 16, 13, 22, 17],
            [14, 14, 13, 19, 15],
            [19, 19, 20, 23, 1000],
            [1000, 0, 1000, 0, 0]
        ])
    ]

    for i, (n_sources, n_destinations, supply, demand, cost) in enumerate(test_data, 1):
        print(f"\nTest Case {i}:")

        problem = TransportationProblem(n_sources, n_destinations, supply, demand, cost)

        print("North-West Corner Method:")
        nw_solution = problem.find_basic_north_west()
        print(nw_solution)
        print("Valid:", is_solution_valid(nw_solution, supply, demand))

        print("Vogel's Approximation Method:")
        vogel_solution = problem.find_basic_vogel()
        print(vogel_solution)
        print("Valid:", is_solution_valid(vogel_solution, supply, demand))

        print("Russell's Approximation Method:")
        russell_solution = problem.find_basic_russel()
        print(russell_solution)
        print("Valid:", is_solution_valid(russell_solution, supply, demand))

# Main function for user input
def main():
    n_sources, n_destinations = map(int, input("Enter number of sources and destinations: ").split())
    supply = list(map(int, input("Enter supply values: ").split()))
    cost = []
    print("Enter cost matrix:")
    for i in range(n_sources):
        cost.append(list(map(int, input().split())))
    demand = list(map(int, input("Enter demand values: ").split()))

    problem = TransportationProblem(n_sources, n_destinations, supply, demand, cost)
    print("North-West:")
    nw_solution = problem.find_basic_north_west()
    print(nw_solution)
    print("Valid:", is_solution_valid(nw_solution, supply, demand))

    print("Vogel:")
    vogel_solution = problem.find_basic_vogel()
    print(vogel_solution)
    print("Valid:", is_solution_valid(vogel_solution, supply, demand))

    print("Russel:")
    russell_solution = problem.find_basic_russel()
    print(russell_solution)
    print("Valid:", is_solution_valid(russell_solution, supply, demand))


if __name__ == '__main__':
    # Run test cases first
    test_cases()
    # Then execute main function for user input
    main()

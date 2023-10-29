class Account:
    # Properties (public)
    holdings: int
    average_price: float
    available_funds: float
    deposited_funds: float
    goal: float
    stop_loss_limit: float

    # Constructor
    def __init__(self, start: float, goal: float, stop_lost_limit: float):
        self.goal = goal
        self.available_funds = start
        self.deposited_funds = start
        self.stop_loss_limit = stop_lost_limit

        self.profit = 0
        self.average_price = 0
        self.holdings = 0

    # Methods
    def calculate_profit(self, current_stock_price: float):
        return round(
            (self.get_total_value(current_stock_price) - self.deposited_funds)
            / self.deposited_funds,
            4,
        )

    def should_exit(self, current_stock_price: float):
        return (
            self.should_trigger_stop_loss(current_stock_price)
            or self.get_total_value(current_stock_price) >= self.goal
        )

    def should_trigger_stop_loss(self, current_stock_price: float):
        return self.get_total_value(current_stock_price) <= self.stop_loss_limit

    def get_total_value(self, current_stock_price: float):
        return self.holdings * current_stock_price + self.available_funds

    def update_holding(self, holding_change: int, fulfilled_price: float):
        # Buy
        if holding_change > 0:
            total_cost = holding_change * fulfilled_price
            new_total_holding_value = (self.holdings * self.average_price) + total_cost
            self.holdings += holding_change
            self.average_price = new_total_holding_value / self.holdings
            self.available_funds -= total_cost

        # Sell
        elif holding_change < 0:
            self.available_funds += abs(holding_change) * fulfilled_price
            self.holdings += holding_change  # holding_change is negative

    def reset(self):
        self.profit = 0
        self.average_price = 0
        self.holdings = 0
        self.available_funds = self.deposited_funds

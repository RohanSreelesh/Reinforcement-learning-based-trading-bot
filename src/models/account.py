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
    def calculate_profit(self):
        return round(
            (self.get_current_account_value() - self.deposited_funds) / self.deposited_funds, 4
        )

    def should_exit(self):
        return self.should_trigger_stop_loss() or self.get_current_account_value() >= self.goal

    def should_trigger_stop_loss(self):
        return self.get_current_account_value() <= self.stop_loss_limit

    def get_current_account_value(self):
        return self.holdings * self.average_price + self.available_funds

    def update_holding(self, holding_change: int, fulfilled_price: float):
        pre_update_holding_value = self.holdings * self.average_price
        pre_update_account_value = self.get_current_account_value()
        self.holdings += holding_change

        # Buy
        if holding_change > 0:
            self.average_price = (
                pre_update_holding_value + holding_change * fulfilled_price
            ) / self.holdings
            self.available_funds -= holding_change * fulfilled_price

        # Sell
        else:
            self.available_funds += holding_change * fulfilled_price

        post_update_acount_value = self.get_current_account_value()

        return round(post_update_acount_value - pre_update_account_value, 2)

    def reset(self):
        self.profit = 0
        self.average_price = 0
        self.holdings = 0
        self.available_funds = self.deposited_funds

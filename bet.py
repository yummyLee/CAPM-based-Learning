import random

count = 1000.0

ratios = 1.96
times = 2
bet_money = 5

count -= bet_money

max_bet = 0

error_count = 0

for i in range(1, 600):

    guess = random.randint(0, 1)

    real = random.randint(0, 1)

    print('real:%d -- guess:%d' % (real, guess))

    if real == guess:
        count += (bet_money * 1.985)
        bet_money = 5
        print('count: %f.2 bet_money: %d' % (count, bet_money))
        count -= bet_money
        error_count = 0
    else:
        error_count += 1
        print('count: %f.2 bet_money: %d' % (count, bet_money))
        # if bet_money <= 40:
        bet_money *= times
        if bet_money > max_bet:
            max_bet = bet_money
        count -= bet_money
        # if count < 0:
        #     print('count < 0: %d' % count)
        #     break

print(max_bet)

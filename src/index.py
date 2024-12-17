import numpy as np
import matplotlib.pyplot as plt
import random

#  ساخت فازی ست مثلثی در بازه های مشخص شده 
def create_fuzzy_sets(start, end, count): 
    step = (end - start) / (count - 1)
    points = np.linspace(start, end, count)
    fuzzy_sets = [(points[i] - step, points[i], points[i] + step) for i in range(len(points))]
    return fuzzy_sets

# تابع عضویت مثلثی دستی
def triangular_MF(x, a, b, c):
# a, b, c: مرزهای مثلث
# x --> ورودی   
# تابع میزان عضویت ورودی x را در فازی ست ها بررسی می کند
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    else:
        return 0.0

# ایجاد فازی ست برای x1, x2 , F
x1_sets = create_fuzzy_sets(-5, 5, 7) # x1 --> [-5,5] , 7 sets
x2_sets = create_fuzzy_sets(-5, 5, 7) # x2 --> [-5,5] , 7 sets
F_sets = create_fuzzy_sets(0, 50, 7) # F --> [0,50] , 7 sets

# دیفازی‌سازی با روش میانگین مراکز
def defuzzify(fuzzy_values, centers):
    """دیفازی‌سازی با روش میانگین مراکز (Center Average)"""
    numerator = sum(mu * center for mu, center in zip(fuzzy_values, centers))  # صورت
    denominator = sum(fuzzy_values)  # مخرج
    return numerator / denominator if denominator != 0 else 0  # جلوگیری از تقسیم بر صفر

# تابع تولید قوانین
def create_fuzzy_rules(train_data):
    rules = []
    for x1, x2, F in train_data:
        # محاسبه درجه عضویت‌ ورودی ها در فازی ست مربوطه
        x1_fuzzy = [triangular_MF(x1, *fs) for fs in x1_sets]
        x2_fuzzy = [triangular_MF(x2, *fs) for fs in x2_sets]
        F_fuzzy = [triangular_MF(F, *fs) for fs in F_sets]  # محاسبه درجه عضویت برای F خروجی

        # انتخاب مجموعه‌هایی که مقادیر در ان ها بیشترین درجه عضویت را دارند
        x1_max = np.argmax(x1_fuzzy)
        x2_max = np.argmax(x2_fuzzy)
        F_max = np.argmax(F_fuzzy)

        # پیدا کردن بیشترین درجات عضویت
        x1_max_value = np.max(x1_fuzzy)
        x2_max_value = np.max(x2_fuzzy)
        F_max_value = np.max(F_fuzzy)

        valid_degree = x1_max_value * x2_max_value * F_max_value # محاسبه درجه اعتبار با ضرب درجه عضویت ورودی ها و خروجی
        rule_degree = min(x1_max_value, x2_max_value, F_max_value) # استفاده از روش min برای ترکیب مقادیر عضویت

        # دیفازی‌سازی (محاسبه مقدار دیفازی‌شده خروجی F)
        defuzzified_output = defuzzify(F_fuzzy, [fs[1] for fs in F_sets])

        # اضافه کردن قانون به لیست قوانین
        rules.append({
            'x1_set': x1_max,
            'x2_set': x2_max,
            'F_set': F_max,
            'rule_degree': rule_degree,
            'valid_degree': valid_degree,
            'defuzzified_output': defuzzified_output
        })  
    return rules, x1_sets, x2_sets, F_sets

def simulated_annealing(train_data, x1_sets, x2_sets, F_sets, initial_rules, max_iter=2000, temp=500, cooling_rate=0.90):
    current_rules = initial_rules
    current_mse, _, _, _ = evaluate_model(train_data, current_rules, x1_sets, x2_sets, F_sets)
    best_rules = current_rules.copy()
    best_mse = current_mse

    for i in range(max_iter):
        temp *= cooling_rate

        new_rules = current_rules.copy()
        rand_idx = random.randint(0, len(new_rules) - 1)
        new_rules[rand_idx]['defuzzified_output'] += np.random.uniform(-0.5, 0.5)

        new_mse, _, _, _ = evaluate_model(train_data, new_rules, x1_sets, x2_sets, F_sets)

        delta_mse = new_mse - current_mse
        if delta_mse < 0 or np.exp(-delta_mse / temp) > random.random():
            current_rules = new_rules
            current_mse = new_mse

        if current_mse < best_mse:
            best_rules = current_rules
            best_mse = current_mse


    return best_rules


# تولید داده‌ها (x1, x2, F) برای train
x1 = np.linspace(-5, 5, 41)
x2 = np.linspace(-5, 5, 41)
x1_grid, x2_grid = np.meshgrid(x1, x2)
F_values = x1_grid**2 + x2_grid**2  # F = x1^2 + x2^2
train_data = np.column_stack((x1_grid.ravel(), x2_grid.ravel(), F_values.ravel()))

# ایجاد قوانین فازی
rules, x1_sets, x2_sets, F_sets = create_fuzzy_rules(train_data)

# یافتن بهترین قانون
optimized_rules = simulated_annealing(train_data, x1_sets, x2_sets, F_sets, rules)

# چاپ قوانین نهایی
print(f"Rules count: {len(optimized_rules)}")
for rule in optimized_rules:
    print(f"Rule: x1 is A{rule['x1_set']} and x2 is B{rule['x2_set']} then F is C{rule['F_set']} "
          f"with rule degree {rule['rule_degree']} and valid degree {rule['valid_degree']:.3f} "
          f"--> Defuzzified output: {rule['defuzzified_output']:.3f}")

# پیش‌ بینی خروجی برای داده‌های تست 
def predict_output(x1, x2, rules, x1_sets, x2_sets, F_sets):
    # محاسبه درجه عضویت‌ داده های تست در فازی ست مربوطه
    x1_fuzzy = [triangular_MF(x1, *fs) for fs in x1_sets]
    x2_fuzzy = [triangular_MF(x2, *fs) for fs in x2_sets]

    # جمع‌آوری خروجی‌های قوانین فعال
    numerator = 0.0
    denominator = 0.0

    for rule in rules:
        # عضویت x1 و x2 در مجموعه‌های مربوط به قانون
        mu_x1 = x1_fuzzy[rule['x1_set']]
        mu_x2 = x2_fuzzy[rule['x2_set']]

        # میزان فعال‌سازی قانون
        rule_activation = min(mu_x1, mu_x2)

        # خروجی دیفازی‌شده این قانون
        rule_output = rule['defuzzified_output']

        # ترکیب خروجی‌ها
        numerator += rule_activation * rule_output
        denominator += rule_activation

    # محاسبه خروجی نهایی دیفازی‌شده
    return numerator / denominator if denominator != 0 else 0.0

# محاسبه MSE و R^2 برای داده‌های تست 
def evaluate_model(test_data, rules, x1_sets, x2_sets, F_sets):
    F_true = []
    F_prediction = []

    for x1, x2, F_actual in test_data:
        # پیش‌بینی خروجی برای هر نقطه تست
        F_estimated = predict_output(x1, x2, rules, x1_sets, x2_sets, F_sets)
        F_true.append(F_actual)
        F_prediction.append(F_estimated)

    # محاسبه MSE
    mse = np.mean((np.array(F_true) - np.array(F_prediction)) ** 2)

    # محاسبه R^2
    ss_total = np.sum((np.array(F_true) - np.mean(F_true)) ** 2)
    ss_residual = np.sum((np.array(F_true) - np.array(F_prediction)) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return mse, r2, F_true, F_prediction

# تولید داده‌های تست
test_x1 = np.linspace(-5, 5, 13)
test_x2 = np.linspace(-5, 5, 13)
test_x1_grid, test_x2_grid = np.meshgrid(test_x1, test_x2)
test_F = test_x1_grid**2 + test_x2_grid**2  # تابع هدف: F = x1^2 + x2^2
test_data = np.column_stack((test_x1_grid.ravel(), test_x2_grid.ravel(), test_F.ravel()))

# ارزیابی مدل
mse, r2, y_true, F_prediction = evaluate_model(test_data, optimized_rules, x1_sets, x2_sets, F_sets)

# نمایش نتایج
print(f"Test MSE: {mse:.4f}")
print(f"Test R^2: {r2:.4f}")

# محاسبه MSE برای داده های تست با تکرار 100
mse_list = []
for _ in range(100):
    mse, r2, F_true, F_prediction = evaluate_model(test_data, optimized_rules, x1_sets, x2_sets, F_sets)
    mse_list.append(mse)

average_mse = np.mean(mse_list)

# نمایش میانگین MSE
print(f"Average MSE over 100 runs: {average_mse:.4f}")

fig = plt.figure(figsize=(12, 6))

# نمودار Train
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_trisurf(train_data[:, 0], train_data[:, 1], train_data[:, 2], cmap='viridis')
ax1.set_title('Train Data: F(x1, x2) = x1^2 + x2^2')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('F(x1, x2)')

# نمودار Test
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(test_data[:, 0], test_data[:, 1], F_prediction , color='r', label='Predicted')
ax2.scatter(test_data[:, 0], test_data[:, 1], F_true, color='g', label='True')
ax2.set_title('Test Data: Predicted vs True Outputs')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('F(x1, x2)')
ax2.legend()

plt.tight_layout()
plt.show()

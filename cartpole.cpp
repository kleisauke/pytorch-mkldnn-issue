#include <torch/nn/modules/linear.h>
#include <torch/optim/adam.h>

class CartPole {
    // Translated from openai/gym's cartpole.py
public:
    double gravity = 9.8;
    double masscart = 1.0;
    double masspole = 0.1;
    double total_mass = (masspole + masscart);
    double length = 0.5; // actually half the pole's length;
    double polemass_length = (masspole * length);
    double force_mag = 10.0;
    double tau = 0.02; // seconds between state updates;

    // Angle at which to fail the episode
    double theta_threshold_radians = 12 * 2 * M_PI / 360;
    double x_threshold = 2.4;
    int steps_beyond_done = -1;

    torch::Tensor state;
    double reward;
    bool done;
    int step_ = 0;

    torch::Tensor getState() {
        return state;
    }

    double getReward() {
        return reward;
    }

    bool isDone() {
        return done;
    }

    void reset() {
        state = torch::empty({4}).uniform_(-0.05, 0.05);
        steps_beyond_done = -1;
        step_ = 0;
    }

    CartPole() : reward(0.0), done(false) {
        reset();
    }

    void step(int action) {
        auto x = state[0].item<float>();
        auto x_dot = state[1].item<float>();
        auto theta = state[2].item<float>();
        auto theta_dot = state[3].item<float>();

        auto force = (action == 1) ? force_mag : -force_mag;
        auto costheta = std::cos(theta);
        auto sintheta = std::sin(theta);
        auto temp = (force + polemass_length * theta_dot * theta_dot * sintheta) /
                    total_mass;
        auto thetaacc = (gravity * sintheta - costheta * temp) /
                        (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        auto xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        x = x + tau * x_dot;
        x_dot = x_dot + tau * xacc;
        theta = theta + tau * theta_dot;
        theta_dot = theta_dot + tau * thetaacc;
        state = torch::tensor({x, x_dot, theta, theta_dot});

        done = x < -x_threshold || x > x_threshold ||
               theta < -theta_threshold_radians || theta > theta_threshold_radians ||
               step_ > 200;

        if (!done) {
            reward = 1.0;
        } else if (steps_beyond_done == -1) {
            // Pole just fell!
            steps_beyond_done = 0;
            reward = 0;
        } else if (steps_beyond_done == 0) {
            assert(false); // Can't do this
        }
        step_++;
    }
};

struct Net : torch::nn::Module {
    Net()
            : linear(4, 128),
              policy(128, 2),
              action(128, 1) {
        register_module("linear", linear);
        register_module("policy", policy);
        register_module("action", action);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor inp) {
        auto x = linear->forward(inp).clamp_min(0);
        torch::Tensor actions = policy->forward(x);
        torch::Tensor value = action->forward(x);
        return std::make_tuple(torch::softmax(actions, -1), value);
    }

    torch::nn::Linear linear;
    torch::nn::Linear policy;
    torch::nn::Linear action;
};

auto main(int argc, const char *argv[]) -> int {
    torch::manual_seed(0);
    std::cerr << "Training episodic policy gradient with a critic for up to 3000"
                 " episodes, rest your eyes for a bit!" << std::endl;
    Net model;
    auto optimizer = torch::optim::Adam(model.parameters(), 1e-3);

    std::vector<torch::Tensor> saved_log_probs;
    std::vector<torch::Tensor> saved_values;
    std::vector<float> rewards;

    auto selectAction = [&](torch::Tensor state) {
        // Only work on single state right now, change index to gather for batch
        auto out = model.forward(state);
        auto probs = torch::Tensor(std::get<0>(out));
        auto value = torch::Tensor(std::get<1>(out));
        auto action = probs.multinomial(1)[0].item<int32_t>();
        // Compute the log prob of a multinomial distribution.
        // This should probably be actually implemented in autogradpp...
        auto p = probs / probs.sum(-1, true);
        auto log_prob = p[action].log();
        saved_log_probs.emplace_back(log_prob);
        saved_values.push_back(value);
        return action;
    };

    auto finishEpisode = [&] {
        auto R = 0.;
        for (int i = rewards.size() - 1; i >= 0; i--) {
            R = rewards[i] + 0.99 * R;
            rewards[i] = R;
        }
        auto r_t = torch::from_blob(
                rewards.data(), {static_cast<int64_t>(rewards.size())});
        r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-5);

        std::vector<torch::Tensor> policy_loss;
        std::vector<torch::Tensor> value_loss;
        for (auto i = 0U; i < saved_log_probs.size(); i++) {
            auto r = rewards[i] - saved_values[i].item<float>();
            policy_loss.push_back(-r * saved_log_probs[i]);
            value_loss.push_back(
                    torch::smooth_l1_loss(saved_values[i], torch::ones(1) * rewards[i]));
        }

        auto loss =
                torch::stack(policy_loss).sum() + torch::stack(value_loss).sum();

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        rewards.clear();
        saved_log_probs.clear();
        saved_values.clear();
    };

    auto env = CartPole();
    double running_reward = 10.0;
    for (size_t episode = 0;; episode++) {
        env.reset();
        auto state = env.getState();
        int t = 0;
        for (; t < 10000; t++) {
            auto action = selectAction(state);
            env.step(action);
            state = env.getState();
            auto reward = env.getReward();
            auto done = env.isDone();

            rewards.push_back(reward);
            if (done) {
                break;
            }
        }

        running_reward = running_reward * 0.99 + t * 0.01;
        finishEpisode();
        if (episode % 10 == 0) {
            std::cout << "Episode " << episode << '\t' << "Last length: " << t << '\t' << "Average length: "
                      << std::setprecision(2) << running_reward << std::endl;
        }
        if (running_reward > 150) {
            break;
        }
        assert(episode < 3000);
    }
}
from mygridworld import GridWorldMDP

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(utility_grids,solver_name):
    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    utility_ssd = np.sum(np.square(np.diff(utility_grids)), axis=(0, 1))
    ax1.plot(utility_ssd, 'b.-')
    ax1.set_ylabel('Change in Utility with {}'.format(solver_name), color='b')

    # policy_changes = np.count_nonzero(np.diff(policy_grids[:,:,:end_step]), axis=(0, 1))
    # ax2.plot(policy_changes, 'r.-')
    # ax2.set_ylabel('Change in Best Policy', color='r')


def plot_difference(ori_utility_grid,utility_grids,solver_name):

    ori_utility_grid = np.expand_dims(ori_utility_grid,axis=-1)
    difference = np.sum(np.square(ori_utility_grid-utility_grids),axis=(0,1))
    plt.plot(difference,'b.-')
    plt.ylabel('Difference with true value grid with {}'.format(solver_name),color='b')


if __name__ == '__main__':
    shape = (4, 4)
    goal_list = [(0,0),(-1,-1)]
    default_reward = -1

    reward_grid = np.zeros(shape) + default_reward
    terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)

    for i in range(len(goal_list)):
        terminal_mask[goal_list[i]] = True

    indices = np.arange(shape[0]*shape[1])
    r0,c0 = np.unravel_index(indices,shape)
    reward_grid[r0[terminal_mask.flatten()],c0[terminal_mask.flatten()]] = 0

    gw = GridWorldMDP(reward_grid=reward_grid,
                      terminal_mask=terminal_mask,
                      action_probabilities=[
                      # the probabilities of action mean possible other steps when taking accurate steps
                          (-1, 0.05),
                          (0, 0.9),
                          (1, 0.05),
                      ],
                      no_action_probability=0.0,
                      accuracy= 1e-5)

    mdp_solvers = {'first_visit_MC':gw.firstv_MC, 'every_visit_MC':gw.everyv_MC,'Temporal_difference':gw.TD_0}
    policy_grids, optimal_utility_grids, end_step = gw.run_policy_iterations(iterations=100, discount=1.0)
    print("Final result of Policy_iteration : ")
    print(optimal_utility_grids[:,:,end_step])
    plt.figure()
    gw.plot_policy(optimal_utility_grids[:,:,end_step])
    plt.savefig("basic_policy_{}_{}.png".format(shape,"policy_iteration"))
    plt.show()
    for solver_name, solver_fn in mdp_solvers.items():

        # plt.figure()
        # plot_convergence(optimal_utility_grids[:,:,:end_step])
        # plt.show()

        utility_grids = solver_fn(iterations = 1000,discount=1.0,policy_grid= policy_grids[:,:,end_step])
        # every_utility_grids = gw.firstv_MC(iterations=1000, discount=1.0, policy_grid=policy_grids[:, :, end_step])
        # utility_grids = np.array(utility_grids)
        print('Final result of {}:'.format(solver_name))
        # utility_grids = solver_fn(discount=1)
        # print(policy_grids[:, :, end_step])
        print(utility_grids[:,:,-1])

        plt.figure()
        gw.plot_policy(utility_grids[:,:,-1])
        plt.savefig('policy_{}_{}.png'.format(shape,solver_name))
        plt.show()

        plot_convergence(utility_grids,solver_name)
        plt.savefig('u&p_change_curve_{}_{}.png'.format(shape,solver_name))
        plt.show()

        plot_difference(optimal_utility_grids[:,:,end_step],utility_grids,solver_name)
        plt.savefig('value_ssd_about_true_value_grid_{}_{}.png'.format(shape,solver_name))
        plt.show()

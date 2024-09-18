def count_last_change(l):
    """
    :param list l: list to be analyzed
    :return: repetitions of the last value l[-1] at the end of the list; e.g., returns 3 for l = [0, 0, 1, 1, 1]
    :rtype: int
    """
    count = 0
    for i in reversed(l):
        if i == l[-1]:
            count += 1
        else:
            break
    return count


def verify_action(config, prev_applied, action, dev):
    """
    :param config: object containing the device parameters
    :param list prev_applied: list containing previous 95 truly applied actions (1 - unblock, 0 - block)
    :param int action: proposed action for the next time step (1 - unblock, 0 - block)
    :param str dev: device type; "wh" or "hp"
    :return:    True - action valid, False - action invalid
                corrected action (1 - unblock, 0 - block)
    :rtype:     bool, int
    """
    if dev == "wh":
        if sum(prev_applied) + action < config.K - config.wh.K_block_day:
            #logging.info("Action invalid: K_block_day_WH exceeded.")
            return False, 1
        else:
            #logging.info("Action valid.")
            return True, action

    elif dev == "hp":
        n_last_change = count_last_change(prev_applied)
        if sum(prev_applied) + action < config.K - config.hp.K_block_day:
            # logging.info("Action invalid: K_block_day_HP exceeded.")
            return False, 1
        elif prev_applied[-1] == 0 and action == 0 and n_last_change >= config.hp.K_block_instance:
            # logging.info("Action invalid: K_block_instance_HP exceeded.")
            return False, 1
        elif prev_applied[-1] == 0 and action == 1 and n_last_change < config.hp.K_min_block:
            # logging.info("Action invalid: K_min_block_HP not fulfilled.")
            return False, 0
        elif prev_applied[-1] == 1 and action == 0 and \
                sum(prev_applied[- (config.K - config.hp.K_min_block):]) < config.K - config.hp.K_block_day:
            # logging.info("Action invalid: If HP is blocked now, either K_min_block_HP or K_block_day_HP has to be violated in the future.")
            return False, 1
        elif prev_applied[-1] == 1 and action == 0 and n_last_change < config.hp.K_min_unblock:
            # logging.info("Action invalid: K_min_unblock_HP not fulfilled.")
            return False, 1
        else:
            # logging.info("Action valid.")
            return True, action

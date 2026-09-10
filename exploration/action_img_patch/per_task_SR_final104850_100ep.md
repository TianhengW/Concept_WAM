# action-as-patch final ckpt (step 104850) — 逐任务 SR, 100 episodes/组

- ckpt: step_104850.pt; 50 任务 x {clean, randomized} x **100 episodes**
- **clean 94.68 / random 94.46 / avg 94.57** (100/100 组)
- 74 组来自权威 100ep run(87165)、26 组为独立重测(87578);未跑满 100ep 的组: 1
  [(('stamp_seal', 'demo_randomized'), 32)]

| # | task | clean SR (ep) | random SR (ep) |
|---|---|---|---|
| 1 | adjust_bottle | 99 (100) | 100 (100) |
| 2 | beat_block_hammer | 98 (100) | 96 (100) |
| 3 | blocks_ranking_rgb | 100 (100) | 99 (100) |
| 4 | blocks_ranking_size | 96 (100) | 95 (100) |
| 5 | click_alarmclock | 98 (100) | 100 (100) |
| 6 | click_bell | 100 (100) | 100 (100) |
| 7 | dump_bin_bigbin | 98 (100) | 95 (100) |
| 8 | grab_roller | 100 (100) | 100 (100) |
| 9 | handover_block | 96 (100) | 98 (100) |
| 10 | handover_mic | 96 (100) | 99 (100) |
| 11 | hanging_mug | 57 (100) | 67 (100) |
| 12 | lift_pot | 100 (100) | 100 (100) |
| 13 | move_can_pot | 100 (100) | 97 (100) |
| 14 | move_pillbottle_pad | 99 (100) | 100 (100) |
| 15 | move_playingcard_away | 100 (100) | 100 (100) |
| 16 | move_stapler_pad | 75 (100) | 71 (100) |
| 17 | open_laptop | 100 (100) | 99 (100) |
| 18 | open_microwave | 96 (100) | 99 (100) |
| 19 | pick_diverse_bottles | 87 (100) | 88 (100) |
| 20 | pick_dual_bottles | 99 (100) | 99 (100) |
| 21 | place_a2b_left | 95 (100) | 97 (100) |
| 22 | place_a2b_right | 98 (100) | 96 (100) |
| 23 | place_bread_basket | 93 (100) | 96 (100) |
| 24 | place_bread_skillet | 87 (100) | 92 (100) |
| 25 | place_burger_fries | 95 (100) | 97 (100) |
| 26 | place_can_basket | 79 (100) | 74 (100) |
| 27 | place_cans_plasticbox | 99 (100) | 97 (100) |
| 28 | place_container_plate | 98 (100) | 100 (100) |
| 29 | place_dual_shoes | 96 (100) | 91 (100) |
| 30 | place_empty_cup | 100 (100) | 100 (100) |
| 31 | place_fan | 95 (100) | 92 (100) |
| 32 | place_mouse_pad | 94 (100) | 93 (100) |
| 33 | place_object_basket | 89 (100) | 83 (100) |
| 34 | place_object_scale | 95 (100) | 98 (100) |
| 35 | place_object_stand | 95 (100) | 94 (100) |
| 36 | place_phone_stand | 100 (100) | 99 (100) |
| 37 | place_shoe | 95 (100) | 98 (100) |
| 38 | press_stapler | 94 (100) | 100 (100) |
| 39 | put_bottles_dustbin | 99 (100) | 94 (100) |
| 40 | put_object_cabinet | 91 (100) | 88 (100) |
| 41 | rotate_qrcode | 94 (100) | 87 (100) |
| 42 | scan_object | 98 (100) | 96 (100) |
| 43 | shake_bottle | 100 (100) | 100 (100) |
| 44 | shake_bottle_horizontally | 100 (100) | 100 (100) |
| 45 | stack_blocks_three | 98 (100) | 99 (100) |
| 46 | stack_blocks_two | 100 (100) | 100 (100) |
| 47 | stack_bowls_three | 83 (100) | 87 (100) |
| 48 | stack_bowls_two | 93 (100) | 95 (100) |
| 49 | stamp_seal | 97 (100) | 93.8 (32) |
| 50 | turn_switch | 90 (100) | 84 (100) |
| — | **AVG** | **94.68** | **94.46** |
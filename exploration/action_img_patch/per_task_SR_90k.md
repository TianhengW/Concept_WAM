# ImageWAM action-as-patch(单流)— 90k ckpt 逐任务 SR

- **ckpt**: `runs/robotwin_flux2_klein_4b_actionpatch_full/2026-08-10_19-24-58/checkpoints/weights/step_090000.pt`
- **eval job**: 86872
- **口径**: RoboTwin 2.0，50 任务 × {clean, randomized} × 10 episodes = 1000 rollouts
- **结果**: clean 94.8 / random 95.0 / **avg 94.9**
- 判 0(HARD TIMEOUT)= 0，无坏数据；50 任务 100/100 单元全部完成。

| # | task | clean SR | random SR | avg |
|---|------|---------:|----------:|----:|
| 1 | adjust_bottle | 100 | 100 | 100.0 |
| 2 | beat_block_hammer | 100 | 100 | 100.0 |
| 3 | blocks_ranking_rgb | 100 | 100 | 100.0 |
| 4 | blocks_ranking_size | 90 | 90 | 90.0 |
| 5 | click_alarmclock | 100 | 100 | 100.0 |
| 6 | click_bell | 100 | 100 | 100.0 |
| 7 | dump_bin_bigbin | 100 | 100 | 100.0 |
| 8 | grab_roller | 100 | 100 | 100.0 |
| 9 | handover_block | 100 | 90 | 95.0 |
| 10 | handover_mic | 100 | 100 | 100.0 |
| 11 | hanging_mug | 30 | 80 | 55.0 |
| 12 | lift_pot | 100 | 100 | 100.0 |
| 13 | move_can_pot | 100 | 100 | 100.0 |
| 14 | move_pillbottle_pad | 100 | 100 | 100.0 |
| 15 | move_playingcard_away | 100 | 100 | 100.0 |
| 16 | move_stapler_pad | 70 | 100 | 85.0 |
| 17 | open_laptop | 100 | 100 | 100.0 |
| 18 | open_microwave | 90 | 50 | 70.0 |
| 19 | pick_diverse_bottles | 100 | 90 | 95.0 |
| 20 | pick_dual_bottles | 100 | 100 | 100.0 |
| 21 | place_a2b_left | 100 | 100 | 100.0 |
| 22 | place_a2b_right | 100 | 100 | 100.0 |
| 23 | place_bread_basket | 100 | 90 | 95.0 |
| 24 | place_bread_skillet | 80 | 100 | 90.0 |
| 25 | place_burger_fries | 90 | 100 | 95.0 |
| 26 | place_can_basket | 60 | 70 | 65.0 |
| 27 | place_cans_plasticbox | 100 | 100 | 100.0 |
| 28 | place_container_plate | 100 | 100 | 100.0 |
| 29 | place_dual_shoes | 90 | 80 | 85.0 |
| 30 | place_empty_cup | 100 | 100 | 100.0 |
| 31 | place_fan | 100 | 100 | 100.0 |
| 32 | place_mouse_pad | 100 | 100 | 100.0 |
| 33 | place_object_basket | 100 | 100 | 100.0 |
| 34 | place_object_scale | 90 | 100 | 95.0 |
| 35 | place_object_stand | 100 | 100 | 100.0 |
| 36 | place_phone_stand | 100 | 100 | 100.0 |
| 37 | place_shoe | 100 | 100 | 100.0 |
| 38 | press_stapler | 100 | 100 | 100.0 |
| 39 | put_bottles_dustbin | 70 | 100 | 85.0 |
| 40 | put_object_cabinet | 100 | 80 | 90.0 |
| 41 | rotate_qrcode | 100 | 80 | 90.0 |
| 42 | scan_object | 100 | 100 | 100.0 |
| 43 | shake_bottle | 100 | 100 | 100.0 |
| 44 | shake_bottle_horizontally | 100 | 100 | 100.0 |
| 45 | stack_blocks_three | 100 | 100 | 100.0 |
| 46 | stack_blocks_two | 100 | 100 | 100.0 |
| 47 | stack_bowls_three | 90 | 70 | 80.0 |
| 48 | stack_bowls_two | 100 | 90 | 95.0 |
| 49 | stamp_seal | 100 | 100 | 100.0 |
| 50 | turn_switch | 90 | 90 | 90.0 |
| — | **AVG (50 tasks)** | **94.8** | **95.0** | **94.9** |
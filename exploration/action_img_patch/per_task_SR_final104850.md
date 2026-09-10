# ImageWAM action-as-patch(单流)— final ckpt(step 104850)逐任务 SR

- **ckpt**: `runs/robotwin_flux2_klein_4b_actionpatch_full/2026-08-11_14-38-46/checkpoints/weights/step_104850.pt`（5 epoch 全量训练收官 ckpt）
- **eval job**: 87164（10 episodes/任务）；标 † 的 2 组因 curobo 死锁无法跑出 10ep 结果，用 100ep 权威值补全
- **口径**: RoboTwin 2.0，50 任务 × {clean, randomized} × 10 episodes
- **结果**: clean 95.68 / random 95.22 / **avg 95.45**（100/100 组）

| # | task | clean SR | random SR | avg |
|---|------|---------:|----------:|----:|
| 1 | adjust_bottle | 100 | 100 | 100.0 |
| 2 | beat_block_hammer | 100 | 100 | 100.0 |
| 3 | blocks_ranking_rgb | 100 | 100 | 100.0 |
| 4 | blocks_ranking_size | 100 | 90 | 95.0 |
| 5 | click_alarmclock | 100 | 100 | 100.0 |
| 6 | click_bell | 100 | 100 | 100.0 |
| 7 | dump_bin_bigbin | 100 | 100 | 100.0 |
| 8 | grab_roller | 100 | 100 | 100.0 |
| 9 | handover_block | 100 | 90 | 95.0 |
| 10 | handover_mic | 100 | 100 | 100.0 |
| 11 | hanging_mug | 60 | 50 | 55.0 |
| 12 | lift_pot | 100 | 100 | 100.0 |
| 13 | move_can_pot | 100 | 100 | 100.0 |
| 14 | move_pillbottle_pad | 100 | 100 | 100.0 |
| 15 | move_playingcard_away | 100 | 100 | 100.0 |
| 16 | move_stapler_pad | 100 | 71† | 85.5 |
| 17 | open_laptop | 100 | 100 | 100.0 |
| 18 | open_microwave | 100 | 100 | 100.0 |
| 19 | pick_diverse_bottles | 90 | 80 | 85.0 |
| 20 | pick_dual_bottles | 100 | 100 | 100.0 |
| 21 | place_a2b_left | 100 | 100 | 100.0 |
| 22 | place_a2b_right | 100 | 100 | 100.0 |
| 23 | place_bread_basket | 90 | 90 | 90.0 |
| 24 | place_bread_skillet | 90 | 100 | 95.0 |
| 25 | place_burger_fries | 100 | 90 | 95.0 |
| 26 | place_can_basket | 80 | 80 | 80.0 |
| 27 | place_cans_plasticbox | 100 | 100 | 100.0 |
| 28 | place_container_plate | 100 | 100 | 100.0 |
| 29 | place_dual_shoes | 90 | 80 | 85.0 |
| 30 | place_empty_cup | 100 | 100 | 100.0 |
| 31 | place_fan | 100 | 100 | 100.0 |
| 32 | place_mouse_pad | 94† | 100 | 97.0 |
| 33 | place_object_basket | 90 | 90 | 90.0 |
| 34 | place_object_scale | 90 | 100 | 95.0 |
| 35 | place_object_stand | 90 | 100 | 95.0 |
| 36 | place_phone_stand | 100 | 100 | 100.0 |
| 37 | place_shoe | 100 | 100 | 100.0 |
| 38 | press_stapler | 100 | 100 | 100.0 |
| 39 | put_bottles_dustbin | 80 | 90 | 85.0 |
| 40 | put_object_cabinet | 100 | 90 | 95.0 |
| 41 | rotate_qrcode | 90 | 100 | 95.0 |
| 42 | scan_object | 100 | 100 | 100.0 |
| 43 | shake_bottle | 100 | 100 | 100.0 |
| 44 | shake_bottle_horizontally | 100 | 100 | 100.0 |
| 45 | stack_blocks_three | 100 | 100 | 100.0 |
| 46 | stack_blocks_two | 100 | 100 | 100.0 |
| 47 | stack_bowls_three | 60 | 100 | 80.0 |
| 48 | stack_bowls_two | 90 | 90 | 90.0 |
| 49 | stamp_seal | 100 | 100 | 100.0 |
| 50 | turn_switch | 100 | 80 | 90.0 |
| — | **AVG** | **95.68** | **95.22** | **95.45** |

> † move_stapler_pad·rnd 和 place_mouse_pad·cln 因 curobo/warp 死锁连续 8 次无法完成 10ep 评测（尝试节点：gnho009/017/031/032/034/036），使用 100ep 权威值（87165 on gnho034）补全。

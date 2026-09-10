"""1-GPU smoke for action_attn_isolate: real weights, forward+backward+infer, and assert the
mask builder actually receives action_len>0 (isolate) / ==0 (original), with img<-/->act invisibility."""
import torch, gc
from exploration.action_img_patch.model import ImageWAMActionPatch
from exploration.action_img_patch import smoke_test as st
from imagewam.models.backbones.imagewam import ImageWAM

FLUX2_MODEL = st.FLUX2_MODEL; AE_MODEL = st.AE_MODEL; FLUX2_SRC = st.FLUX2_SRC; QWEN3 = st.QWEN3
REC = []
_orig = ImageWAM._build_mot_attention_mask_flux2
def _spy(self, **kw):
    out = _orig(self, **kw)
    REC.append((kw["txt_len"], kw["cond_len"], kw["target_len"], kw["action_len"], out["double_joint"][0].clone().cpu()))
    return out
ImageWAM._build_mot_attention_mask_flux2 = _spy

def build(isolate: bool):
    return ImageWAMActionPatch.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=FLUX2_MODEL, ae_model_path=AE_MODEL, flux2_src_path=FLUX2_SRC,
        qwen3_model_spec=QWEN3, qwen_context_len=128, proprio_dim=14, load_text_encoder=True,
        device="cuda", torch_dtype=torch.bfloat16, action_patch_dim=14, action_attn_isolate=isolate)

def check_mask(rec, isolate, tag):
    txt, cond, tl, al, mk = rec
    x0 = txt + cond
    if isolate:
        S, H = tl, al; img = slice(x0, x0+S); act = slice(x0+S, x0+S+H)
        assert al == 16, f"[{tag}] isolate 但 action_len={al}"
        assert not mk[img, act].any() and not mk[act, img].any(), f"[{tag}] isolate 下 img/act 仍互见"
    else:
        assert al == 0, f"[{tag}] 非 isolate 但 action_len={al}"
        S = tl - 16; img = slice(x0, x0+S); act = slice(x0+S, x0+tl)
        assert mk[img, act].all() and mk[act, img].all(), f"[{tag}] 原 AP 下 img/act 不再互见"
    assert not mk[:x0, x0:].any(), f"[{tag}] clean 看到了 noisy"
    ref = slice(txt, x0); tx = slice(0, txt)
    ii, aa = bool(mk[img, img].all()), bool(mk[act, act].all())
    a_ref, i_ref = bool(mk[act, ref].all()), bool(mk[img, ref].all())
    same_txt = torch.equal(mk[act, tx].all(0), mk[img, tx].all(0))   # action 与 image 看到同一组有效 text 列
    n_txt_valid = int(mk[img, tx].all(0).sum())
    print(f"  [{tag}] img自注意={ii} act自注意={aa} act->ref={a_ref} img->ref={i_ref} act/img看到相同text列={same_txt} 有效text列={n_txt_valid}/{txt}(其余为padding)")
    assert ii and aa and a_ref and i_ref and same_txt and n_txt_valid > 0, f"[{tag}] 结构断言失败"
    print(f"  [{tag}] mask 收到 txt={txt} cond={cond} target_len={tl} action_len={al} -> img<->act 互见={bool(mk[img,act].any())}  ✔")

for isolate in (True, False):
    tag = f"isolate={isolate}"
    REC.clear()
    model = build(isolate)
    assert model.action_attn_isolate is isolate
    st.apply_trainer_freeze(model)
    st.check_backward(model, tag)              # training_loss + backward + finite + grads
    assert REC, "训练路径没有调用 mask 构造函数"
    check_mask(REC[-1], isolate, tag + " train")
    REC.clear()
    with torch.no_grad():
        out = model.infer_action(prompt="pick up the bottle", input_image=torch.rand(1,3,288,256),
                                 action_horizon=16, proprio=torch.randn(1,14), num_inference_steps=2)
    assert out["action"].shape == (16, 14)
    assert len(REC) == 1, f"推理应只建一次 mask,实际 {len(REC)}"
    check_mask(REC[-1], isolate, tag + " infer")
    del model; gc.collect(); torch.cuda.empty_cache()
print("SMOKE_ISOATTN PASSED")

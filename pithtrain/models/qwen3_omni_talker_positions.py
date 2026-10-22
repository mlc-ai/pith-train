"""Qwen3-Omni Talker prompt T/H/W positions.

The public Transformers 5.17 position algorithm is retained here under its
Apache-2.0 license. This pure tensor builder has no HF modeling dependency.
User/assistant token IDs belong to the Thinker vocabulary and are deliberately
not validated against the smaller Talker codec vocabulary.
"""

import torch

from pithtrain.models.qwen3_omni_audio import audio_output_lengths


class TalkerPositionBuilder:
    def __init__(self, config, *, audio_n_window=50):
        if audio_n_window != 50:
            raise ValueError("Talker positions support the released audio n_window=50")
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.audio_n_window = audio_n_window

    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[torch.Tensor],
        grid_hs: list[torch.Tensor],
        grid_ws: list[torch.Tensor],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = (
            torch.arange(llm_grid_h)
            .view(1, -1, 1)
            .expand(len(t_index), -1, llm_grid_w)
            .flatten()
            .float()
        )
        w_index = (
            torch.arange(llm_grid_w)
            .view(1, 1, -1)
            .expand(len(t_index), llm_grid_h, -1)
            .flatten()
            .float()
        )
        t_index = (
            torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
        )
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
        audio_seqlens: torch.LongTensor | None = None,
        second_per_grids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Talker prompt IDs must be a nonempty [batch, time] tensor")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Talker prompt IDs must be integers")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if (
            attention_mask.shape != input_ids.shape
            or not ((attention_mask == 0) | (attention_mask == 1)).all()
        ):
            raise ValueError("Talker prompt mask must be binary and match input_ids")
        if not (attention_mask.sum(-1) > 0).all():
            raise ValueError("Every Talker prompt must contain at least one unmasked token")
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = (0, 0, 0)
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i]]
                image_nums, video_nums, audio_nums = (0, 0, 0)
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = (image_nums, video_nums, audio_nums)
                multimodal_nums = (
                    image_nums + audio_nums
                    if use_audio_in_video
                    else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                        remain_videos > 0 or remain_images > 0
                    ):
                        ed_vision_start = input_tokens.index(vision_start_token_id, st)
                    else:
                        ed_vision_start = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio_start = input_tokens.index(audio_start_token_id, st)
                    else:
                        ed_audio_start = len(input_tokens) + 1
                    min_ed = min(ed_vision_start, ed_audio_start)
                    text_len = min_ed - st
                    if text_len != 0:
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )
                        st_idx += text_len
                    if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        bos_len, eos_len = (2, 2)
                    else:
                        bos_len, eos_len = (1, 1)
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    st_idx += bos_len
                    if min_ed == ed_audio_start:
                        audio_len = audio_output_lengths(
                            audio_seqlens[audio_idx], self.audio_n_window
                        )
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)
                        st += int(text_len + bos_len + audio_len + eos_len)
                        audio_idx += 1
                        remain_audios -= 1
                    elif (
                        min_ed == ed_vision_start
                        and input_ids[ed_vision_start + 1] == image_token_id
                    ):
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // spatial_merge_size**2
                        llm_pos_ids_list.append(llm_pos_ids)
                        st += int(text_len + bos_len + image_len + eos_len)
                        image_idx += 1
                        remain_images -= 1
                    elif (
                        min_ed == ed_vision_start
                        and input_ids[ed_vision_start + 1] == video_token_id
                    ):
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t)
                            * second_per_grids[video_idx].cpu().float()
                            * position_id_per_seconds
                        ).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // spatial_merge_size**2
                        llm_pos_ids_list.append(llm_pos_ids)
                        st += int(text_len + bos_len + video_len + eos_len)
                        video_idx += 1
                        remain_videos -= 1
                    elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        audio_len = audio_output_lengths(
                            audio_seqlens[audio_idx], self.audio_n_window
                        )
                        audio_llm_pos_ids = (
                            torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        )
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t)
                            * second_per_grids[video_idx].cpu().float()
                            * position_id_per_seconds
                        ).float()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_data_index, audio_data_index = (0, 0)
                        while (
                            video_data_index < video_llm_pos_ids.shape[-1]
                            and audio_data_index < audio_llm_pos_ids.shape[-1]
                        ):
                            if (
                                video_llm_pos_ids[0][video_data_index]
                                <= audio_llm_pos_ids[0][audio_data_index]
                            ):
                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_data_index : video_data_index + 1]
                                )
                                video_data_index += 1
                            else:
                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1]
                                )
                                audio_data_index += 1
                        if video_data_index < video_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]]
                            )
                        if audio_data_index < audio_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]]
                            )
                        video_len = video_grid_thw[video_idx].prod() // spatial_merge_size**2
                        st += int(text_len + bos_len + audio_len + video_len + eos_len)
                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )
                llm_positions = torch.cat(
                    [item.float() for item in llm_pos_ids_list], dim=1
                ).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return (position_ids, mrope_position_deltas)
        else:
            position_ids = attention_mask.float().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = (
                max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
            )
            return (position_ids, mrope_position_deltas)

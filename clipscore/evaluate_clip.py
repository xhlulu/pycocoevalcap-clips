import clip
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.args = args
        self.data = data
        self.preprocess = self._transform_test(args.input_resolution)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, preprocess, args, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions, args.prefix),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(args.device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, preprocess, args, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images, args),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(args.device)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features

    

class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, fnameids, batch_size=32):
        self.fnameids = fnameids
        self.all_idxs = np.array(range(len(self.fnameids)))
        self.batch_size = batch_size
        self.transform = self._transform()

    def __len__(self):
        return math.ceil(len(self.fnameids) / self.batch_size)

    def _transform(self, n_px=224):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def load_and_preprocess(self, img_name):

        h = self.transform(Image.open(img_name))
        return h.numpy()

    def __getitem__(self, idx):
        batch_idxs = self.all_idxs[idx * self.batch_size:
                                   (idx + 1) * self.batch_size]
        all_fnames = [self.fnameids[idx] for idx in batch_idxs]
        all_img_arrs = []
        for f in all_fnames:
            all_img_arrs.append(self.load_and_preprocess(f))
        res = torch.Tensor(np.stack(all_img_arrs, axis=0))
        return res


class CaptionSequence(tf.keras.utils.Sequence):
    def __init__(self, captions, batch_size=128, prefix=None):
        self.captions = captions
        self.all_idxs = np.array(range(len(self.captions)))
        self.batch_size = batch_size
        if prefix is None:
            #self.prefix = 'A picture of '
            self.prefix = 'A photo depicts '
        else:
            self.prefix = prefix

    def clip_tokenize(self, texts, context_length=77):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = clip.clip._tokenizer.encoder["<|startoftext|>"]
        eot_token = clip.clip._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [clip.clip._tokenizer.encode(text)[:context_length-2] for text in texts]
        all_tokens = [[sot_token] + toks + [eot_token] for toks in all_tokens]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                print(f"Input {texts[i]} is too long for context length {context_length}")
                print('Truncating from {} --> {}'.format(len(tokens), context_length))
                tokens = tokens[:context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result

    def __len__(self):
        return math.ceil(len(self.captions) / self.batch_size)

    def __getitem__(self, idx):
        batch_idxs = self.all_idxs[idx * self.batch_size:
                                   (idx + 1) * self.batch_size]

        to_get = self.clip_tokenize([self.prefix + ' '.join(self.captions[idx].split()) for idx in batch_idxs])
        return to_get



def get_clip_scores_simple(images, candidates, prefix=None, w=2.5):
    caption_sequence = CaptionSequence(
        candidates,
        batch_size=128,
        prefix=prefix)

    image_sequence = ImageSequence(
        images,
        batch_size=128)

    image_feats = []
    with torch.no_grad():
        for batch in tqdm.tqdm(image_sequence):
            batch = batch.to(device=device)
            if device == 'cuda':
                batch = batch.to(torch.float16)
            image_feats.append(model.encode_image(batch).cpu().numpy())

    image_feats = np.vstack(image_feats)

    caption_feats = []
    with torch.no_grad():
        for batch in tqdm.tqdm(caption_sequence):
            batch = batch.to(device=device)
            caption_feats.append(model.encode_text(batch).cpu().numpy())
    caption_feats = np.vstack(caption_feats)

    image_feats = sklearn.preprocessing.normalize(image_feats, axis=1)
    caption_feats = sklearn.preprocessing.normalize(caption_feats, axis=1)

    per = w*np.clip(np.sum(image_feats * caption_feats, axis=1), 0, None)
    return np.mean(per), per


def get_refonlyclipscore_dot(candidates, references, prefix=None):
    candidate_sequence = CaptionSequence(
        candidates,
        batch_size=128,
        prefix=prefix
    )

    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    refs_sequence = CaptionSequence(
        flattened_refs,
        batch_size=128,
        prefix=prefix
    )

    candidate_feats = []
    with torch.no_grad():
        for batch in tqdm.tqdm(candidate_sequence):
            batch = batch.to(device=device)
            candidate_feats.append(model.encode_text(batch).cpu().numpy())
    candidate_feats = np.vstack(candidate_feats)

    reference_feats = []
    with torch.no_grad():
        for batch in tqdm.tqdm(refs_sequence):
            batch = batch.to(device=device)
            reference_feats.append(model.encode_text(batch).cpu().numpy())
    reference_feats = np.vstack(reference_feats)

    candidate_feats = sklearn.preprocessing.normalize(candidate_feats, axis=1)
    reference_feats = sklearn.preprocessing.normalize(reference_feats, axis=1)

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(reference_feats, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidate_feats)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    method = 'max'
    for c_idx, cand in tqdm.tqdm(enumerate(candidate_feats)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        if method == 'max':
            per.append(np.max(all_sims))
        elif method == 'mean':
            per.append(np.mean(all_sims))

    return np.mean(per), per

# refclipscore = 2 * clip_per * refclip_per / (clip_per + refclip_per)

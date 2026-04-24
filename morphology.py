from dataclasses import dataclass, field
from typing import List


@dataclass
class MorphInfo:
    token: str
    stem: str
    affixes: List[str] = field(default_factory=list)
    case_markers: List[str] = field(default_factory=list)
    tam_markers: List[str] = field(default_factory=list)
    morph_flags: List[int] = field(default_factory=list)


class HeuristicMorphSegmenter:
    '''
    Placeholder heuristic segmenter for Manipuri/Bengali-script Manipuri.
    Replace these suffix lists with your curated morphology rules.
    '''

    CASE_MARKERS = [
        "না", "দা", "গী", "গা", "বু", "তা", "দি", "কি"
    ]
    TAM_MARKERS = [
        "খ্রে", "খি", "নি", "বা", "উ", "ল্লি", "রে", "গনি"
    ]

    def segment(self, token: str) -> MorphInfo:
        affixes = []
        case_markers = []
        tam_markers = []
        stem = token

        for suf in sorted(self.CASE_MARKERS, key=len, reverse=True):
            if token.endswith(suf) and len(token) > len(suf):
                affixes.append(suf)
                case_markers.append(suf)
                stem = token[:-len(suf)]
                break

        for suf in sorted(self.TAM_MARKERS, key=len, reverse=True):
            if token.endswith(suf) and len(token) > len(suf):
                affixes.append(suf)
                tam_markers.append(suf)
                if stem == token:
                    stem = token[:-len(suf)]
                break

        morph_flags = [
            1 if case_markers else 0,
            1 if tam_markers else 0,
            len(affixes),
            len(token),
            1 if token[:1].isupper() else 0,
            1 if any(ch.isdigit() for ch in token) else 0,
            1 if len(token) > 6 else 0,
            1 if token.endswith("া") else 0,
        ]

        return MorphInfo(
            token=token,
            stem=stem,
            affixes=affixes,
            case_markers=case_markers,
            tam_markers=tam_markers,
            morph_flags=morph_flags,
        )

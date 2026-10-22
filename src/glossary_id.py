# -*- coding: utf-8 -*-
"""
glossary_id.py
~~~~~~~~~~~~~~
The Bahasa Indonesia edition of the plain-language vocabulary in ``glossary``.

The keys match ``glossary.TERMS`` exactly, so a page asks for a term and gets
the wording back in the reader's language. ``term`` is the word as it should
be printed: most sports words are translated, but "pit stop", "sprint",
"stint", "pace", "undercut" and "H2H" keep the form Indonesian broadcasts and
paddock reporting use, because translating them would confuse a reader who
has heard the English word.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

from typing import Dict

TERMS: Dict[str, Dict[str, str]] = {
    "race": {
        "term": "Balapan",
        "short": "Balapan utama hari Minggu. Poinnya dihitung untuk kejuaraan.",
        "long": "Grand Prix adalah acara utama akhir pekan balapan. Setiap pembalap "
                "mendapat poin sesuai urutan finis, dan pembalap serta tim dengan poin "
                "terbanyak di akhir musim menjadi juara.",
    },
    "sprint": {
        "term": "Sprint",
        "short": "Balapan pendek hari Sabtu. Memberi poin, tetapi lebih sedikit "
                 "daripada balapan utama.",
        "long": "Sprint adalah balapan yang lebih pendek (sekitar 100 km) pada hari "
                "Sabtu. Sprint memberi poin kejuaraan, tetapi lebih sedikit daripada "
                "Grand Prix, dan tidak menentukan posisi start untuk balapan.",
    },
    "qualifying": {
        "term": "Kualifikasi",
        "short": "Sesi singkat hari Sabtu yang menentukan siapa start di posisi mana.",
        "long": "Kualifikasi adalah sesi berwaktu pada hari Sabtu. Lap tercepat "
                "menentukan urutan start: pembalap tercepat start pertama (pole "
                "position), tercepat kedua start kedua, dan seterusnya.",
    },
    "grid": {
        "term": "Grid",
        "short": "Urutan start balapan, ditentukan oleh kualifikasi.",
        "long": "Grid adalah urutan mobil saat balapan dimulai. Urutannya berasal "
                "dari kualifikasi, dan menjadi patokan pembanding ramalan di aplikasi "
                "ini.",
    },
    "points": {
        "term": "Poin",
        "short": "Yang diperebutkan pembalap dan tim. Menang bernilai paling besar.",
        "long": "Poin diberikan untuk setiap posisi finis di balapan dan sprint. "
                "Pembalap dengan poin terbanyak menjadi juara dunia pembalap, tim "
                "dengan poin terbanyak menjadi juara dunia konstruktor.",
    },
    "stint": {
        "term": "Stint",
        "short": "Rentetan lap seorang pembalap dengan satu set ban.",
        "long": "Stint adalah rentetan lap yang diselesaikan dengan satu set ban "
                "sebelum pit stop. Biasanya ada tiga stint dalam satu balapan: yang "
                "pertama dengan ban start, yang tengah paling panjang, dan yang "
                "terakhir dengan ban baru.",
    },
    "pit stop": {
        "term": "Pit stop",
        "short": "Berhenti untuk mengganti ban. Memakan beberapa detik, jadi hanya "
                 "layak bila ban sudah aus.",
        "long": "Pit stop adalah berhenti sebentar di pit lane untuk mengganti ban. "
                "Satu pit stop normal memakan beberapa detik waktu di lintasan, jadi "
                "pembalap berhenti hanya ketika keausan ban membuat waktu yang hilang "
                "lebih murah daripada cengkeraman yang hilang.",
    },
    "degradation": {
        "term": "Degradasi",
        "short": "Seberapa cepat ban kehilangan cengkeraman saat dipakai.",
        "long": "Degradasi ban adalah laju hilangnya performa ban saat digunakan. "
                "Semakin lunak ban, semakin cepat aus: ban lunak cepat tetapi berumur "
                "pendek, ban keras lebih lambat tetapi tahan banyak lap.",
    },
    "undercut": {
        "term": "Undercut",
        "short": "Masuk pit beberapa lap lebih awal daripada lawan untuk mendapat "
                 "ban lebih segar lalu menyalip.",
        "long": "Undercut adalah strategi: masuk pit beberapa lap sebelum lawan, "
                "melaju dengan ban lebih segar, lalu kembali di depan. Kebalikannya - "
                "bertahan lebih lama di luar - disebut overcut.",
    },
    "delta": {
        "term": "Delta",
        "short": "Selisih waktu: dengan mobil di depan, atau dengan lap tercepat.",
        "long": "Delta adalah selisih waktu. Di layar waktu, angkanya menunjukkan "
                "jarak waktu dengan mobil di depan, atau selisih dengan lap tercepat "
                "sesi ini.",
    },
    "sector": {
        "term": "Sektor",
        "short": "Sepertiga sirkuit. Membandingkan waktu sektor menunjukkan di mana "
                 "seorang pembalap lebih cepat.",
        "long": "Sirkuit dibagi menjadi tiga sektor. Waktu tiap sektor dicatat "
                "terpisah, jadi waktu sektor dua pembalap menunjukkan di mana salah "
                "satunya mendapat atau kehilangan waktu.",
    },
    "rolling average": {
        "term": "Rata-rata bergerak",
        "short": "Rata-rata tiga hasil terakhir. Lebih stabil daripada satu balapan.",
        "long": "Rata-rata bergerak menghaluskan tiga hasil terakhir menjadi satu "
                "angka. Angka ini menggambarkan tren performa, bukan satu balapan yang "
                "beruntung atau sial, dan angka lebih kecil lebih baik.",
    },
    "pace": {
        "term": "Pace",
        "short": "Seberapa cepat satu mobil dalam satu lap setelah lap keluar pit dan "
                 "lalu lintas dibuang.",
        "long": "Pace adalah waktu lap yang bisa dijaga seorang pembalap. Diukur hanya "
                "pada lap bersih: lap keluar dari pit, lap di belakang safety car, dan "
                "lap yang ditandai tidak akurat tidak dihitung.",
    },
    "estimated": {
        "term": "Perkiraan",
        "short": "Dihitung dari data, bukan diukur langsung.",
        "long": "Semua angka di aplikasi ini yang diturunkan (bukan dilaporkan "
                "langsung) ditandai perkiraan. Angka itu disertai tingkat keyakinan "
                "sesuai banyaknya lap bersih yang menjadi dasarnya.",
    },
    "confidence": {
        "term": "Keyakinan",
        "short": "Seberapa banyak data yang menjadi dasar sebuah perkiraan.",
        "long": "Tingkat keyakinan mengikuti ukuran sampel: baik berarti minimal 15 lap "
                "bersih, sedang minimal 8, rendah minimal 2. Di bawah itu aplikasi "
                "menulis data tidak cukup daripada menebak.",
    },
    "backtest": {
        "term": "Uji mundur",
        "short": "Model dijalankan pada balapan lampau, hanya belajar dari seri "
                 "sebelumnya, lalu dibandingkan dengan patokan sederhana.",
        "long": "Uji mundur mengulang ramalan pada balapan satu musim. Untuk tiap "
                "balapan, model hanya melihat seri sebelumnya, jadi tidak bisa curang "
                "dengan melihat hasil yang diramalnya. Hasilnya dibandingkan dengan "
                "patokan yang sekadar memakai urutan start.",
    },
    "baseline": {
        "term": "Patokan",
        "short": "Pembanding sederhana: yang start pertama finis pertama.",
        "long": "Patokan ini menganggap urutan start menentukan urutan finis. Model "
                "baru layak dipakai bila mengalahkannya, dan aplikasi menyebut terus "
                "terang bila belum.",
    },
    "pit window": {
        "term": "Jendela pit",
        "short": "Rentang lap ketika pit stop diperkirakan menguntungkan.",
        "long": "Jendela pit adalah rentang lap ketika pembalap kehilangan lebih "
                "sedikit dengan pit stop daripada bertahan dengan ban aus. Aplikasi "
                "mencarinya, bukan mengasumsikan rencana tetap.",
    },
    "replay": {
        "term": "Replay",
        "short": "Tampilan animasi balapan: di mana setiap mobil berada di lintasan.",
        "long": "Replay memperlihatkan posisi setiap mobil sepanjang sesi, "
                "diperkirakan dari feed posisi publik. Ini replay posisi, bukan waktu "
                "resmi atau video on-board.",
    },
    "setup fingerprint": {
        "term": "Sidik setup",
        "short": "Perkiraan kasar setelan mobil dari cara mobil menikung.",
        "long": "Sidik setup adalah perkiraan, bukan lembar setelan: sirkuit dibagi "
                "menjadi beberapa tikungan dan perilaku mobil di tiap tikungan "
                "dibandingkan untuk menebak apakah mobil disetel untuk downforce "
                "tinggi atau rendah. Semua ini perkiraan dari telemetri publik.",
    },
    "h2h": {
        "term": "H2H",
        "short": "Head to head: rekor dua pembalap saat dibandingkan langsung.",
        "long": "Rekor head-to-head hanya menghitung balapan yang diikuti kedua "
                "pembalap. Ini cara paling adil membandingkan dua pembalap, karena "
                "cuaca, sirkuit, dan aturannya sama bagi keduanya.",
    },
    "podium": {
        "term": "Podium",
        "short": "Tiga finis teratas sebuah balapan. Poin podium adalah hadiah "
                 "terbesar dalam satu balapan.",
        "long": "Podium adalah tiga posisi finis teratas. Pembalap di podium "
                "mengumpulkan poin terbesar: 25 untuk juara, 18 untuk kedua, dan 15 "
                "untuk ketiga.",
    },
    "championship": {
        "term": "Kejuaraan",
        "short": "Gelar juara: siapa pun dengan poin terbanyak saat musim berakhir.",
        "long": "Ada dua kejuaraan. Kejuaraan pembalap untuk pembalap dengan poin "
                "terbanyak; kejuaraan konstruktor untuk tim yang dua pembalapnya "
                "mengumpulkan poin terbanyak bersama.",
    },
    "fastest lap": {
        "term": "Lap tercepat",
        "short": "Satu lap tercepat dalam sesi. Memberi satu poin tambahan.",
        "long": "Lap tercepat adalah lap tercepat dalam sebuah balapan. Pembalap yang "
                "mencetaknya mendapat satu poin tambahan, bila finis di sepuluh besar.",
    },
    "tyre": {
        "term": "Ban",
        "short": "Karet yang dipakai mobil. Ban lunak cepat tetapi cepat aus.",
        "long": "Ban tersedia dalam kompon lunak, sedang, dan keras. Ban lunak memberi "
                "cengkeraman paling besar untuk beberapa lap lalu aus; ban keras tahan "
                "sepanjang balapan tetapi lebih lambat. Pembalap berganti ban di pit.",
    },
    "constructor": {
        "term": "Konstruktor",
        "short": "Tim yang mengikutkan mobil. Dua pembalap mengumpulkan poin untuk "
                 "satu tim.",
        "long": "Konstruktor adalah tim di balik mobil - Mercedes, Ferrari, Red Bull "
                "Racing. Dua pembalapnya berbagi poin konstruktor, jadi tim bisa juara "
                "walau salah satu pembalapnya mengumpulkan poin lebih sedikit daripada "
                "pembalap tim lain.",
    },
    "clean lap": {
        "term": "Lap bersih",
        "short": "Lap yang layak dibandingkan: tanpa pit lane, tanpa safety car, "
                 "tanpa kesalahan.",
        "long": "Lap bersih adalah lap dengan waktu sah di lintasan hijau. Lap keluar "
                "pit, lap di belakang safety car, dan lap yang ditandai tidak akurat "
                "oleh sistem waktu tidak dihitung, karena tidak mewakili kecepatan "
                "mobil.",
    },
    "forecast": {
        "term": "Ramalan",
        "short": "Perkiraan hasil balapan berikutnya, disusun dari urutan start, "
                 "performa tim terakhir, dan hasil pembalap.",
        "long": "Ramalan ini model yang terbuka, bukan kotak hitam: regresi atas "
                "urutan start, pace tim musim ini, hasil terakhir pembalap, posisi "
                "yang diperoleh di lintasan, dan keandalan finis. Ramalan diuji mundur "
                "pada balapan lampau untuk melihat apakah lebih baik daripada sekadar "
                "menyalin urutan start, dan aplikasi menyebut terus terang bila belum.",
    },
    "risk": {
        "term": "Risiko",
        "short": "Seberapa mungkin ramalan itu salah.",
        "long": "Risiko berasal dari seberapa dekat angkanya: bila dua pembalap hampir "
                "sama dalam skor model, salah satu bisa finis di depan yang lain, dan "
                "ramalan mengatakan hal itu daripada berpura-pura urutannya pasti.",
    },
    "margin": {
        "term": "Selisih",
        "short": "Jarak waktu juara dengan pembalap kedua, dalam detik.",
        "long": "Selisih kemenangan adalah waktu antara pembalap pertama dan kedua "
                "saat finis. Selisih beberapa persepuluh detik berarti hasilnya "
                "ditentukan di lap terakhir.",
    },
}

function depth (target) {
    var max = 0,
        childs = target.querySelectorAll("*");
    if (childs.length === 0)
        return 1;
    for (var i = 0; i < childs.length; i++) {
        child_depth = depth(childs[i]);
        if (child_depth > max)
            max = child_depth;
    };
    return 1 + max;
}
console.log(depth(document.body) + 2);
document.querySelectorAll("*").length;
